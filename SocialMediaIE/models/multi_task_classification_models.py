import logging
from typing import Dict, Iterable, Iterator, List, Optional, Union, Set, Any


from allennlp.common import Params, Registrable
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.models.model import Model
from allennlp.modules.text_field_embedders import (
    TextFieldEmbedder,
    BasicTextFieldEmbedder,
)
from allennlp.modules.seq2vec_encoders import (
    Seq2VecEncoder,
    PytorchSeq2VecWrapper,
    CnnEncoder,
    CnnHighwayEncoder,
    BagOfEmbeddingsEncoder
)

from allennlp.modules.seq2seq_encoders import (
    Seq2SeqEncoder,
    PytorchSeq2SeqWrapper,
    StackedSelfAttentionEncoder,
    PassThroughEncoder,
    IntraSentenceAttentionEncoder,
)
from allennlp.data.iterators import BucketIterator, DataIterator
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.basic_classifier import BasicClassifier
from allennlp.training.metrics import CategoricalAccuracy

import torch
import tqdm
import os
import logging

from SocialMediaIE.training.multitask_trainer import Task
from SocialMediaIE.evaluation.metrics import FBetaMeasure
from SocialMediaIE.models.helpers import AllenNLPSequential

logger = logging.getLogger(__name__)


def get_encoder(input_dim, output_dim, encoder_type, args):
    if encoder_type == "bag":
        return BagOfEmbeddingsEncoder(input_dim)
    if encoder_type == "bilstm":
        return PytorchSeq2VecWrapper(
            AllenNLPSequential(
                torch.nn.ModuleList([
                    get_encoder(input_dim, output_dim, "bilstm-unwrapped", args)
                ]),
                input_dim,
                output_dim,
                bidirectional=True,
                residual_connection=args.residual_connection,
                dropout=args.dropout
            )
        )
    if encoder_type == "bilstm-unwrapped":
        return torch.nn.LSTM(
            input_dim,
            output_dim,
            batch_first=True,
            bidirectional=True,
            dropout=args.dropout,
        )
    if encoder_type == "cnn":
        return CnnEncoder(
            embedding_dim=input_dim,
            num_filters=output_dim
        )
    if encoder_type == "cnn_highway":
        filter_size: int = output_dim//4
        return CnnHighwayEncoder(
            embedding_dim=input_dim,
            filters=[
                (2, filter_size),
                (3, filter_size),
                (4, filter_size),
                (5, filter_size)
            ],
            projection_dim=output_dim,
            num_highway=3,
            do_layer_norm=True,
        )
    raise RuntimeError(f"Unknown encoder type={encoder_type}")


def get_task_encoder_dict(args, task_suffixes: Set[str], elmo_embedding_dim: int):
    hidden_dim = args.hidden_dim
    dropout = args.dropout
    encoder_type = args.encoder_type
    residual_connection = args.residual_connection
    def get_wrapped_encoder(encoder_list):
        return PytorchSeq2VecWrapper(
            AllenNLPSequential(
                torch.nn.ModuleList(encoder_list),
                elmo_embedding_dim,
                hidden_dim,
                bidirectional=True,
                residual_connection=residual_connection,
                dropout=dropout
            )
        )
    if len(task_suffixes) == 1:
        single_task_suffix = tuple(task_suffixes)[0]
        encoders = torch.nn.ModuleDict(
            {single_task_suffix: get_encoder(elmo_embedding_dim, hidden_dim, encoder_type, args)}
        )
    else:
        # More than one type of task
        multi_task_mode = args.multi_task_mode
        if multi_task_mode == "shared":
            shared_encoder = get_encoder(elmo_embedding_dim, hidden_dim, encoder_type, args)
            encoders = torch.nn.ModuleDict(
                {suffix: shared_encoder for suffix in task_suffixes}
            )
        elif multi_task_mode == "invidual":
            # should be same as using full task models here for comparison
            encoders = torch.nn.ModuleDict(
                {
                    suffix: get_encoder(elmo_embedding_dim, hidden_dim, encoder_type, args)
                    for suffix in task_suffixes
                }
            )
        elif multi_task_mode == "stacked":
            assert task_suffixes == set(
                ["sentiment", "abusive", "uncertainity"]
            ), f"All tasks should be present. Found {task_suffixes}"
            logger.info(
                f"For multi_task_mode={multi_task_mode}, ignoring encoder_type={args.encoder_type}. All encoders are biLSTM"
            )
            INTERMEDIATE_INPUT = 2 * hidden_dim
            encoder_type = "bilstm-unwrapped"
            sentiment_encoder = get_encoder(elmo_embedding_dim, hidden_dim, encoder_type, args)
            abusive_encoder = get_encoder(INTERMEDIATE_INPUT, hidden_dim, encoder_type, args)
            uncertainity_encoder = get_encoder(INTERMEDIATE_INPUT, hidden_dim, encoder_type, args)
            encoder_list = torch.nn.ModuleList([
                sentiment_encoder, abusive_encoder, uncertainity_encoder
            ])
            encoders = torch.nn.ModuleDict(
                {
                    "sentiment": get_wrapped_encoder(encoder_list[:1]),
                    "abusive": get_wrapped_encoder(encoder_list[:2]),
                    "uncertainity": get_wrapped_encoder(encoder_list),
                }
            )
        elif multi_task_mode == "stacked-allennlp":
            DEFAULT_TASK_ORDER=["sentiment", "abusive", "uncertainity"]
            task_order = DEFAULT_TASK_ORDER
            if hasattr(args, "task_order"):
                task_order = args.task_order
            task_order = [k for k in task_order if k in task_suffixes]
            n_tasks = len(task_order)
        else:
            raise RuntimeError(f"multi_task_mode={multi_task_mode} not implemented.")
    return encoders

class BasicClassifierWithMetrics(BasicClassifier):
    def __init__(self, *args, **kwargs):
        super(BasicClassifierWithMetrics, self).__init__(*args, **kwargs)
        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "fbeta": FBetaMeasure(average="micro")
        }

    def forward(self, tokens, **kwargs):
        label = kwargs["label"]
        output_dict = super(BasicClassifierWithMetrics, self).forward(tokens, label)
        logits = output_dict["logits"]
        if label is not None:
            for metric in self.metrics.values():
                metric(logits, label)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for metric_name, metric in self.metrics.items():
            metric_value = metric.get_metric(reset)
            if isinstance(metric_value, dict):
                for k,v in metric_value.items():
                    if isinstance(v, list):
                        metrics.update({f"k_{i}": vi for i, vi in enumerate(v)})
                    else:
                        metrics[k] = v
            else:
                metrics[metric_name] = metric_value
        return metrics

class MultiTaskClassifier(Model):
    def __init__(
        self,
        word_embeddings: TextFieldEmbedder,
        encoders: torch.nn.ModuleDict,
        vocab: Vocabulary,
        tasks: List[Task],
    ) -> None:
        super().__init__(vocab)
        self._tasks = tasks
        self.word_embeddings = word_embeddings
        self.encoders = encoders
        # self.hidden2tag = dict()
        self.classifier = torch.nn.ModuleDict()
        self.metrics = dict()
        self._inference_mode = False
        for task in tasks:
            tag_namespace = task.tag_namespace
            self.classifier[tag_namespace] = BasicClassifierWithMetrics(
                vocab=vocab,
                text_field_embedder=self.word_embeddings,
                seq2vec_encoder=self.encoders[task.task_type],
                label_namespace=tag_namespace,
            )

    def set_inference_mode(self, mode=True):
        """Only compute the CRF for all tasks if inference mode is one. 
        Otherwise computing CRF for each task during training and evaluation is waste.
        CRF computation is N^2 operation and hence slows the training."""
        self._inference_mode = mode

    def get_task_tags(self, inputs, tag_namespace_indicies, tag_namespace):
        tag_namespace_idx = self.vocab.get_token_index(tag_namespace, "label_namespace")
        tag_namespace_batch_indicies = (
            (tag_namespace_indicies == tag_namespace_idx).nonzero().squeeze(-1)
        )
        tags = None
        if tag_namespace_batch_indicies.shape[0] > 0:
            tags = inputs[tag_namespace][tag_namespace_batch_indicies]
        return tags

    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        tag_namespace_indicies = inputs["label_namespace"]
        output = {}
        output["loss"] = 0.0
        for task in self._tasks:
            # Find valid task output indicies
            tag_namespace = task.tag_namespace
            tags = self.get_task_tags(inputs, tag_namespace_indicies, tag_namespace)
            if tags is None:
                classifier_outputs = {}  # Set as empty dict
                if self._inference_mode:
                    classifier_outputs = self.classifier[tag_namespace].forward(
                        label=tags, **inputs
                    )
                    output[f"{tag_namespace}_probs"] = classifier_outputs["probs"]
            else:
                # Only compute CRF when tag is not none.
                classifier_outputs = self.classifier[tag_namespace].forward(label=tags, **inputs)
                output[f"{tag_namespace}_probs"] = classifier_outputs["probs"]

            if "loss" in classifier_outputs and tags is not None:
                num_instances = tags.shape[0]
                output[f"{tag_namespace}_loss"] = classifier_outputs["loss"] / num_instances
                output["loss"] += output[f"{tag_namespace}_loss"]
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for task in self._tasks:
            tag_namespace = task.tag_namespace
            classifier_metrics = self.classifier[tag_namespace].get_metrics(reset)
            for metric_name, metric_value in classifier_metrics.items():
                metrics[f"{tag_namespace}_{metric_name}"] = metric_value
        return metrics
