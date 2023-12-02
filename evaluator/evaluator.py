"""
Adapted from https://github.com/allenai/ms2 and
https://github.com/allenai/mslr-shared-task/
Copied from Evidence Inference
"""
import itertools
import os
import re
from typing import List, Optional, Dict
from dataclasses import dataclass

import json
import warnings
import numpy as np


import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pad_sequence,
    PackedSequence,
    pack_padded_sequence,
    pad_packed_sequence,
)
from datasets import load_dataset, Dataset
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    PretrainedConfig,
    AutoTokenizer,
)
import bert_score
from rouge_score import rouge_scorer, scoring
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
from scipy.spatial.distance import jensenshannon


START_POPULATION = "<pop>"
END_POPULATION = "</pop>"
START_INTERVENTION = "<int>"
END_INTERVENTION = "</int>"
START_OUTCOME = "<out>"
END_OUTCOME = "</out>"
START_BACKGROUND = "<background>"
END_BACKGROUND = "</background>"
START_REFERENCE = "<ref>"
END_REFERENCE = "</ref>"
START_EVIDENCE = "<evidence>"
END_EVIDENCE = "</evidence>"
SEP_TOKEN = "<sep>"
EXTRA_TOKENS = [
    START_BACKGROUND,
    END_BACKGROUND,
    START_REFERENCE,
    END_REFERENCE,
    SEP_TOKEN,
    START_POPULATION,
    END_POPULATION,
    START_INTERVENTION,
    END_INTERVENTION,
    START_OUTCOME,
    END_OUTCOME,
    START_EVIDENCE,
    END_EVIDENCE,
]
INTERVENTION_RE = START_INTERVENTION + "(.*?)" + END_INTERVENTION
OUTCOME_RE = START_OUTCOME + "(.*?)" + END_OUTCOME

DATASET: Dict[str, Dataset] = {}

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.set_device(0)
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


@dataclass(eq=True, frozen=True)
class PaddedSequence:
    """A utility class for padding variable length sequences mean for RNN input
    This class is in the style of PackedSequence from the PyTorch RNN Utils,
    but is somewhat more manual in approach. It provides the ability to generate masks
    for outputs of the same input dimensions.
    The constructor should never be called directly and should only be called via
    the autopad classmethod.
    We'd love to delete this, but we pad_sequence, pack_padded_sequence, and
    pad_packed_sequence all require shuffling around tuples of information, and some
    convenience methods using these are nice to have.
    """

    data: torch.Tensor
    batch_sizes: torch.Tensor
    batch_first: bool = False

    @classmethod
    def autopad(
        cls, data, batch_first: bool = False, padding_value=0, device=None
    ) -> "PaddedSequence":
        # handle tensors of size 0 (single item)
        data_ = []
        for d in data:
            if len(d.size()) == 0:
                d = d.unsqueeze(0)
            data_.append(d)
        padded = pad_sequence(
            data_, batch_first=batch_first, padding_value=padding_value
        )
        if batch_first:
            batch_lengths = torch.LongTensor([x.size()[0] for x in data_])
            if any([x == 0 for x in batch_lengths]):
                raise ValueError(
                    "Found a 0 length batch element, this can't possibly be right: {}".format(
                        batch_lengths
                    )
                )
        else:
            # TODO actually test this codepath
            batch_lengths = torch.LongTensor([len(x) for x in data])
        return PaddedSequence(padded, batch_lengths, batch_first).to(device=device)

    def pack_other(self, data: torch.Tensor):
        return pack_padded_sequence(
            data, self.batch_sizes, batch_first=self.batch_first, enforce_sorted=False
        )

    @classmethod
    def from_packed_sequence(
        cls, ps: PackedSequence, batch_first: bool, padding_value=0
    ) -> "PaddedSequence":
        padded, batch_sizes = pad_packed_sequence(ps, batch_first, padding_value)
        return PaddedSequence(padded, batch_sizes, batch_first)

    def cuda(self) -> "PaddedSequence":
        return PaddedSequence(
            self.data.cuda(), self.batch_sizes.cuda(), batch_first=self.batch_first
        )

    def to(
        self, device=None, dtype=None, copy=False, non_blocking=False
    ) -> "PaddedSequence":
        # TODO make to() support all of the torch.Tensor to() variants
        return PaddedSequence(
            self.data.to(
                device=device, dtype=dtype, copy=copy, non_blocking=non_blocking
            ),
            self.batch_sizes.to(device=device, copy=copy, non_blocking=non_blocking),
            batch_first=self.batch_first,
        )

    def mask(
        self, on=int(0), off=int(0), device="cpu", size=None, dtype=None
    ) -> torch.Tensor:
        if size is None:
            size = self.data.size()
        out_tensor = torch.zeros(*size, dtype=dtype)
        # TODO this can be done more efficiently
        out_tensor.fill_(off)
        # note to self: these are probably less efficient than explicilty populating the off values instead of the on values.
        if self.batch_first:
            for i, bl in enumerate(self.batch_sizes):
                out_tensor[i, :bl] = on
        else:
            for i, bl in enumerate(self.batch_sizes):
                out_tensor[:bl, i] = on
        return out_tensor.to(device)

    def unpad(self, other: torch.Tensor = None) -> List[torch.Tensor]:
        if other is None:
            other = self
        if isinstance(other, PaddedSequence):
            other = other.data
        out = []
        for o, bl in zip(other, self.batch_sizes):
            out.append(o[:bl])
        return out

    def flip(self) -> "PaddedSequence":
        return PaddedSequence(
            self.data.transpose(0, 1), not self.batch_first, self.padding_value
        )


def initialize_models(params: dict, unk_token="<unk>"):
    max_length = params["max_length"]
    tokenizer = RobertaTokenizer.from_pretrained(params["bert_vocab"])
    # tokenizer = BertTokenizer.from_pretrained(params['bert_vocab'])
    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    evidence_classes = dict(
        (y, x) for (x, y) in enumerate(params["evidence_classifier"]["classes"])
    )
    if bool(params.get("random_init", 0)):
        with open(params["bert_config"], "r") as inf:
            cfg = inf.read()
            id_config = PretrainedConfig.from_dict(json.loads(cfg), num_labels=2)
            cls_config = PretrainedConfig.from_dict(
                json.loads(cfg), num_labels=len(evidence_classes)
            )
        use_half_precision = bool(
            params["evidence_identifier"].get("use_half_precision", 0)
        )
        evidence_identifier = BertClassifier(
            bert_dir=None,
            pad_token_id=pad_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            num_labels=2,
            max_length=max_length,
            use_half_precision=use_half_precision,
            config=id_config,
        )
        use_half_precision = bool(
            params["evidence_classifier"].get("use_half_precision", 0)
        )
        evidence_classifier = BertClassifier(
            bert_dir=None,
            pad_token_id=pad_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            num_labels=len(evidence_classes),
            max_length=max_length,
            use_half_precision=use_half_precision,
            config=cls_config,
        )
    else:
        bert_dir = params["bert_dir"]
        use_half_precision = bool(
            params["evidence_identifier"].get("use_half_precision", 0)
        )
        evidence_identifier = BertClassifier(
            bert_dir=bert_dir,
            pad_token_id=pad_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            num_labels=2,
            max_length=max_length,
            use_half_precision=use_half_precision,
        )
        use_half_precision = bool(
            params["evidence_classifier"].get("use_half_precision", 0)
        )
        evidence_classifier = BertClassifier(
            bert_dir=bert_dir,
            pad_token_id=pad_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            num_labels=len(evidence_classes),
            max_length=max_length,
            use_half_precision=use_half_precision,
        )
    word_interner = tokenizer.get_vocab()
    de_interner = dict((x, y) for (y, x) in word_interner.items())
    # de_interner = tokenizer.ids_to_tokens
    return (
        evidence_identifier,
        evidence_classifier,
        word_interner,
        de_interner,
        evidence_classes,
        tokenizer,
    )


class BertClassifier(nn.Module):
    """Thin wrapper around BertForSequenceClassification"""

    def __init__(
        self,
        bert_dir: Optional[str],
        pad_token_id: int,
        cls_token_id: int,
        sep_token_id: int,
        num_labels: int,
        max_length: int = 512,
        use_half_precision=False,
        config: Optional[PretrainedConfig] = None,
    ):
        super(BertClassifier, self).__init__()
        if bert_dir is None:
            assert config is not None
            assert config.num_labels == num_labels
            bert = RobertaForSequenceClassification(config)
            # bert = BertForSequenceClassification(config)
        else:
            bert = RobertaForSequenceClassification.from_pretrained(
                bert_dir, num_labels=num_labels
            )
            # bert = BertForSequenceClassification.from_pretrained(bert_dir, num_labels=num_labels)
        if use_half_precision:
            import apex

            bert = bert.half()
        self.bert = bert
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.max_length = max_length

    def forward(self, query: List[torch.tensor], document_batch: List[torch.tensor]):
        assert len(query) == len(document_batch)
        # note about device management:
        # since distributed training is enabled, the inputs to this module can be on *any* device (preferably cpu, since we wrap and unwrap the module)
        # we want to keep these params on the input device (assuming CPU) for as long as possible for cheap memory access
        target_device = next(self.parameters()).device
        cls_token = torch.tensor(
            [self.cls_token_id]
        )  # .to(device=document_batch[0].device)
        sep_token = torch.tensor(
            [self.sep_token_id]
        )  # .to(device=document_batch[0].device)
        input_tensors = []
        position_ids = []
        for q, d in zip(query, document_batch):
            if len(q) + len(d) + 2 > self.max_length:
                d = d[: (self.max_length - len(q) - 2)]
            input_tensors.append(
                torch.cat([cls_token, q, sep_token, d.to(dtype=q.dtype)])
            )
            position_ids.append(torch.arange(0, input_tensors[-1].size().numel()))
            # position_ids.append(torch.tensor(list(range(0, len(q) + 1)) + list(range(0, len(d) + 1))))
        bert_input = PaddedSequence.autopad(
            input_tensors,
            batch_first=True,
            padding_value=self.pad_token_id,
            device=target_device,
        )
        positions = PaddedSequence.autopad(
            position_ids, batch_first=True, padding_value=0, device=target_device
        )
        outputs = self.bert(
            bert_input.data,
            attention_mask=bert_input.mask(
                on=1.0, off=0.0, dtype=torch.float, device=target_device
            ),
            position_ids=positions.data,
        )
        (classes,) = outputs.logits  # had to fix this bug from MSLR
        assert torch.all(classes == classes)  # for nans
        return classes


def ios(preamble):
    # we know that the reference abstract is already space tokenized
    start_stop_words = EXTRA_TOKENS + ["<s>", "</s>"]

    def clean_str(s):
        for w in start_stop_words:
            s = s.replace(w, "")
        return s

    outcomes = list(map(clean_str, re.findall(OUTCOME_RE, preamble)))
    interventions = list(map(clean_str, re.findall(INTERVENTION_RE, preamble)))
    return interventions, outcomes


def evidence_inference_score(
    model, evidence_inference_tokenizer, summary, preamble, use_ios
):
    ret = []
    if use_ios:
        interventions, outcomes = ios(preamble)
        summary = evidence_inference_tokenizer(summary, return_tensors="pt")[
            "input_ids"
        ]
        for i, o in itertools.product(interventions, outcomes):
            preamble = i + " " + evidence_inference_tokenizer.sep_token + " " + o
            ico = evidence_inference_tokenizer(preamble, return_tensors="pt")[
                "input_ids"
            ]
            classes = model(ico, summary)
            classes = torch.softmax(classes, dim=-1).detach().cpu().squeeze().tolist()
            significance_distribution = dict(
                zip(
                    [
                        "significantly decreased",
                        "no significant difference",
                        "significantly increased",
                    ],
                    classes,
                )
            )
            ret.append(significance_distribution)
    else:
        preamble = ""
        ico = evidence_inference_tokenizer(preamble, return_tensors="pt")["input_ids"]
        summary = evidence_inference_tokenizer(summary, return_tensors="pt")[
            "input_ids"
        ]
        classes = model(ico, summary)
        classes = torch.softmax(classes, dim=-1).detach().cpu().squeeze().tolist()
        significance_distribution = dict(
            zip(
                [
                    "significantly decreased",
                    "no significant difference",
                    "significantly increased",
                ],
                classes,
            )
        )
        ret.append(significance_distribution)

    return ret


def jsd(m1, m2):
    keys = list(set(m1.keys()) | set(m2.keys()))
    m1 = [m1.get(k, 0) for k in keys]
    m2 = [m2.get(k, 0) for k in keys]
    return jensenshannon(m1, m2, base=2)


def entailment_score(
    model, evidence_inference_tokenizer, generated, target, preamble, use_ios
):
    generated_distributions = evidence_inference_score(
        model, evidence_inference_tokenizer, generated, preamble, use_ios
    )
    summary_distributions = evidence_inference_score(
        model, evidence_inference_tokenizer, target, preamble, use_ios
    )
    jsds = []
    for generated_distribution, summary_distribution in zip(
        generated_distributions, summary_distributions
    ):
        jsds.append(jsd(generated_distribution, summary_distribution))
    if len(jsds) == 0:
        return None
    return np.mean(jsds)


def f1_score(
    model, evidence_inference_tokenizer, generateds, targets, preambles, use_ios
):
    summary_preds = []
    generated_preds = []
    in_doc_classifications = []
    labels = [
        "significantly decreased",
        "no significant difference",
        "significantly increased",
    ]
    mapping = {x: i for (i, x) in enumerate(labels)}
    for generated, target, preamble in zip(generateds, targets, preambles):
        generated_distributions = evidence_inference_score(
            model, evidence_inference_tokenizer, generated, preamble, use_ios
        )
        summary_distributions = evidence_inference_score(
            model, evidence_inference_tokenizer, target, preamble, use_ios
        )
        in_doc_generated = []
        in_doc_target = []
        for generated_distribution, summary_distribution in zip(
            generated_distributions, summary_distributions
        ):
            generated_targets = sorted(
                generated_distribution.items(), key=lambda x: x[1]
            )
            summary_targets = sorted(summary_distribution.items(), key=lambda x: x[1])
            best_summary_target = summary_targets[-1][0]
            in_doc_target.append(best_summary_target)
            summary_preds.append(best_summary_target)
            generated_target = generated_targets[-1][0]
            generated_preds.append(generated_target)
            in_doc_generated.append(generated_target)
        if len(in_doc_generated) == 0:
            continue
        # surpress UndefinedMetricWarning warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            in_doc_classifications.append(
                classification_report(
                    np.array([mapping[x] for x in in_doc_target]),
                    np.array([mapping[x] for x in in_doc_generated]),
                    target_names=labels,
                    labels=list(range(len(labels))),
                    output_dict=True,
                    digits=4,
                )
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        res = classification_report(
            np.array([mapping[x] for x in summary_preds]),
            np.array([mapping[x] for x in generated_preds]),
            target_names=labels,
            output_dict=True,
            labels=list(range(len(labels))),
            digits=4,
        )
    return res


def jsd_uniform(model, evidence_inference_tokenizer, target, preamble, use_ios):
    summary_distributions = evidence_inference_score(
        model, evidence_inference_tokenizer, target, preamble, use_ios
    )
    jsds = []
    # baseline distributions
    generated_distribution = {
        "significantly decreased": 0.134,
        "no significant difference": 0.570,
        "significantly increased": 0.296,
    }
    for summary_distribution in summary_distributions:
        jsds.append(jsd(generated_distribution, summary_distribution))
    if len(jsds) == 0:
        return None
    return np.mean(jsds)


def entailment_scores(
    model, evidence_inference_tokenizer, generateds, targets, preambles, use_ios
):
    """
    Compute the entailment score for a list of generated summaries
    :param model: the EI model
    :param evidence_inference_tokenizer: the EI tokenizer
    :param generateds: the generated summaries
    :param targets: the target summaries
    :param preambles: the preambles (can be empty)
    :param use_ios: whether to use Intervention Outcome pairs
    """
    f1_scores = f1_score(
        model, evidence_inference_tokenizer, generateds, targets, preambles, use_ios
    )
    scores = list(
        map(
            lambda x: entailment_score(
                model, evidence_inference_tokenizer, *x, use_ios
            ),
            zip(generateds, targets, preambles),
        )
    )
    scores = list(filter(lambda x: x is not None, scores))
    uniform_scores = list(
        map(
            lambda x: jsd_uniform(model, evidence_inference_tokenizer, *x, use_ios),
            zip(targets, preambles),
        )
    )
    uniform_scores = list(filter(lambda x: x is not None, uniform_scores))
    assert len(scores) > 0
    avg = np.mean(scores)
    s = np.std(scores)
    uniform_score = np.mean(uniform_scores)
    return {
        "average": avg,
        "std": s,
        "uniform_preds": uniform_score,
        "f1_score": f1_scores,
        "scores": scores,
    }


def clean(s):
    for t in EXTRA_TOKENS + ["<s>", "</s>"]:
        s = s.replace(t, "")
        s = s.replace("  ", " ")
    return s


class MS2Evaluator:
    def __init__(
        self,
        bertscore_model_type="microsoft/deberta-xlarge-mnli",
        device=None,
    ):
        """
        :param model_type: model type for BERTscore. Choose from many choices (look
            up bert-score github). Note that MSLR/MS2 uses roberta-large in default
            args, but https://huggingface.co/allenai/led-base-16384-ms2 uses
            `microsoft/deberta-xlarge-mnli`. Note that `bert-score`
            recommends `microsoft/deberta-xlarge-mnli` as the one with best
            correlation with human judgement. Details here:
            https://huggingface.co/microsoft/deberta-xlarge-mnli
        """
        self.rouge_tokenizer: Optional[AutoTokenizer] = None
        self.bertscore_model_type = bertscore_model_type
        self.evidence_inference_params: Optional[dict] = None
        self.evidence_inference_tokenizer: Optional[RobertaTokenizer] = None
        self.evidence_inference_classifier: Optional[BertClassifier] = None
        self.evidence_classifier: Optional[BertClassifier] = None
        self.device = device or DEVICE
        self.results: Optional[dict] = None

        self.evidence_inference_param_file = os.path.join(
            os.path.dirname(__file__), "bert_pipeline_8samples.json"
        )
        self.evidence_inference_model_dir = os.path.join(
            os.path.dirname(__file__), "evidence_inference_models"
        )

    def rouge_scores(
        self,
        preds: List[List[torch.Tensor]],
        targets: List[List[torch.Tensor]],
        tokenizer,
        use_stemmer=False,
        use_aggregator=False,
    ) -> Dict:
        # largely copied from https://github.com/huggingface/nlp/blob/master/metrics/rouge/rouge.py#L84
        # and from https://github.com/allenai/ms2/blob/a03ab009e00c5e412b4c55f6ec4f9b49c2d8a7f6/ms2/models/utils.py
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        scorer = rouge_scorer.RougeScorer(
            rouge_types=rouge_types, use_stemmer=use_stemmer
        )
        refs, hyps = [], []
        for p, t in zip(preds, targets):
            assert len(p) == len(t)
            refs.extend(p)
            hyps.extend(t)

        if use_aggregator:
            aggregator = scoring.BootstrapAggregator()
            scores = None
        else:
            aggregator = None
            scores = []

        for ref, pred in zip(refs, hyps):
            if isinstance(ref, torch.Tensor):
                ref = tokenizer.decode(ref).lower()
            if isinstance(pred, torch.Tensor):
                pred = tokenizer.decode(pred).lower()
            score = scorer.score(ref, pred)
            if use_aggregator:
                aggregator.add_scores(score)
            else:
                scores.append(score)

        if use_aggregator:
            result = aggregator.aggregate()
        else:
            result = {}
            for key in scores[0]:
                result[key] = list(score[key] for score in scores)

        return result

    def get_tokenizer(self, tokenizer_type: str):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_type, additional_special_tokens=EXTRA_TOKENS
        )
        return tokenizer

    def calculate_rouge(
        self, targets: Dict[str, Dict], generated: Dict[str, str]
    ) -> Dict:
        """
        Calculate ROUGE scores
        :param targets: dict of docid -> {'target': target_text}
        :param generated: dict of docid -> generated_text
        :return: dict of ROUGE scores (rouge1, rouge2, rougeL, rougeLsum)
        """
        # copied from https://github.com/allenai/mslr-shared-task/blob/c2218c1a440cf5172d784065b48af2d6c5c50f9a/evaluator/evaluator.py
        print("Computing ROUGE scores...")
        docids = list(targets.keys())
        target_texts = [[targets[docid]["target"]] for docid in docids]
        generated_texts = [[generated.get(docid, "")] for docid in docids]

        # rouge scoring
        if self.rouge_tokenizer is None:
            self.rouge_tokenizer = self.get_tokenizer("facebook/bart-base")

        rouge_results = self.rouge_scores(
            generated_texts, target_texts, self.rouge_tokenizer, use_aggregator=True
        )
        self.results["rouge"] = rouge_results
        return rouge_results

    def calculate_mid_rouge(
        self, targets: Dict[str, Dict], generated: Dict[str, str]
    ) -> Dict:
        """
        Calculate ROUGE scores but only return mid F1.
        :param targets: dict of docid -> {'target': target_text}
        :param generated: dict of docid -> generated_text
        :return: dict of ROUGE scores (rouge1, rouge2, rougeL, rougeLsum)
        """
        results = self.calculate_rouge(targets, generated)
        ret = {
            "rouge1": results["rouge1"].mid.fmeasure,
            "rouge2": results["rouge2"].mid.fmeasure,
            "rougeL": results["rougeL"].mid.fmeasure,
            "rougeLsum": results["rougeLsum"].mid.fmeasure,
        }
        self.results.update(ret)
        return ret

    def calculate_bertscore(
        self,
        targets: Dict[str, Dict],
        generated: Dict[str, str],
    ) -> Dict:
        """
        Calculate BERTscore
        :param targets: dict of docid -> {'target': target_text}
        :param generated: dict of docid -> generated_text
        :return: dict of BERTscore results (bertscore_precisions, bertscore_recalls, bertscore_f1s) (precision, recall, f1)
        """
        # copied from https://github.com/allenai/mslr-shared-task/blob/c2218c1a440cf5172d784065b48af2d6c5c50f9a/evaluator/evaluator.py
        # original bert score: https://github.com/Tiiiger/bert_score
        print("Computing BERTscore...")
        docids = list(targets.keys())
        target_texts = [targets[docid]["target"] for docid in docids]
        generated_texts = [generated.get(docid, "") for docid in docids]

        # BERTscore
        bertscore_precisions, bertscore_recalls, bertscore_f1s = bert_score.score(
            generated_texts, target_texts, model_type=self.bertscore_model_type
        )
        ret = {
            "bertscore_precisions": bertscore_precisions,
            "bertscore_recalls": bertscore_recalls,
            "bertscore_f1s": bertscore_f1s,
        }
        self.results.update(ret)
        return ret

    def calculate_mean_bertscore(
        self,
        targets: Dict[str, Dict],
        generated: Dict[str, str],
    ) -> Dict:
        """
        Calculate mean BERTscore
        :param targets: dict of docid -> {'target': target_text}
        :param generated: dict of docid -> generated_text
        :param model_type: model type for BERTscore.
        :return: dict of mean BERTscore results (bertscore_precisions, bertscore_recalls, bertscore_f1s) (precision, recall, f1)
        """
        individual_results = self.calculate_bertscore(targets, generated)

        results = {
            "bertscore_avg_precision": torch.mean(
                individual_results["bertscore_precisions"]
            ).item(),
            "bertscore_avg_recall": torch.mean(
                individual_results["bertscore_recalls"]
            ).item(),
            "bertscore_avg_f1": torch.mean(individual_results["bertscore_f1s"]).item(),
            "bertscore_std_precision": torch.std(
                individual_results["bertscore_precisions"]
            ).item(),
            "bertscore_std_recall": torch.std(
                individual_results["bertscore_recalls"]
            ).item(),
            "bertscore_std_f1": torch.std(individual_results["bertscore_f1s"]).item(),
        }
        self.results.update(results)
        return results

    def calculate_evidence_inference_divergence(
        self,
        targets: Dict[str, Dict],
        generated: Dict[str, str],
        evidence_inference_use_unconditional: bool = False,
    ) -> Dict:
        """
        Calculate Evidence Inference Divergence
        :param targets: dict of docid -> {'target': target_text , 'preface': preface_text}
        :param generated: dict of docid -> generated_text
        :param ei_use_unconditional: [True/False] whether to use unconditional
            evidence inference model
        :return:
        """
        print("Computing Delta Evidence Inference scores...")
        docids = list(targets.keys())
        target_texts = [targets[docid]["target"] for docid in docids]
        preface_texts = [targets[docid]["background"] for docid in docids]
        generated_texts = [generated.get(docid, "") for docid in docids]

        generated_texts = list(map(clean, generated_texts))
        target_texts = list(map(clean, target_texts))

        # evidence inference scoring
        with open(self.evidence_inference_param_file, "r") as inf:
            self.evidence_inference_params = json.loads(inf.read())

        if (
            self.evidence_inference_tokenizer is None
            or self.evidence_inference_classifier is None
        ):
            (
                _,
                self.evidence_inference_classifier,
                _,
                _,
                _,
                self.evidence_inference_tokenizer,
            ) = initialize_models(self.evidence_inference_params)
            if evidence_inference_use_unconditional:
                classifier_file = os.path.join(
                    self.evidence_inference_model_dir,
                    "unconditioned_evidence_classifier",
                    "unconditioned_evidence_classifier.pt",
                )
            else:
                classifier_file = os.path.join(
                    self.evidence_inference_model_dir,
                    "evidence_classifier",
                    "evidence_classifier.pt",
                )

            # pooler parameters are added by default in an older transformers,
            # so we have to ignore that those are uninitialized.
            self.evidence_inference_classifier.load_state_dict(
                torch.load(classifier_file, map_location=self.device), strict=False
            )
            if self.device.type == "cuda":
                self.evidence_inference_classifier.cuda()
            elif self.device.type == "mps":
                self.evidence_inference_classifier.to(self.device)

        entailment_results = entailment_scores(
            self.evidence_inference_classifier,
            self.evidence_inference_tokenizer,
            generated_texts,
            target_texts,
            preface_texts,
            use_ios=evidence_inference_use_unconditional,
        )
        self.results["evidence_inference_divergence"] = entailment_results
        return entailment_results

    def load_data(self, split="validation"):
        """
        Load data from MSLR dataset
        :param split: which split to load
        :return: None
        """
        global DATASET
        if split not in DATASET:
            DATASET[split] = load_dataset("allenai/mslr2022", "ms2", split=split)

        return DATASET[split]

    def evaluate(
        self,
        generated: Dict[str, str],
        targets: Optional[Dict[str, Dict]] = None,
        split="validation",
        evidence_inference_use_unconditional: bool = False,
    ) -> Dict:
        """
        Evaluate generated summaries
        :param generated: dict of docid -> generated_text
        :param targets: dict of docid -> {'target': target_text , 'preface': preface_text}. If None, load from MSLR dataset based on `split`
        :param split: which split to load. Only used if `targets` is None
        :param evidence_inference_use_unconditional: [True/False] whether to use unconditional
            evidence inference model
        :return: dict of evaluation results
        """
        self.results = {}
        # ensure that generated keys are str, not int
        generated = {str(k): v for k, v in generated.items()}

        if targets is None:
            dataset = self.load_data(split=split)
            targets = {row["review_id"]: row for row in dataset}

            # subset targets to only ones that have generated summaries
            targets = {k: v for k, v in targets.items() if k in generated}

        else:
            # ensure that target keys are str, not int
            targets = {str(k): v for k, v in targets.items()}

        # assert that the sets are exactly the same
        assert set(generated.keys()) == set(targets.keys())
        print(f"Number of generated summaries: {len(generated)}")

        self.results["generated"] = generated
        self.results["targets"] = targets

        self.calculate_rouge(targets, generated)
        self.calculate_mean_bertscore(targets, generated)
        self.calculate_evidence_inference_divergence(
            targets, generated, evidence_inference_use_unconditional
        )

        return self.results
