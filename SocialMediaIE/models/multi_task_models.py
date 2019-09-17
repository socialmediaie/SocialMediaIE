import logging
from typing import Dict, Iterable, Iterator, List, Optional, Union, Set, Any


from allennlp.common import Params, Registrable
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.models.model import Model
from allennlp.modules.text_field_embedders import (
    TextFieldEmbedder,
    BasicTextFieldEmbedder,
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
from allennlp.models.crf_tagger import CrfTagger


import torch
import tqdm
import os
import logging

from SocialMediaIE.training.multitask_trainer import Task
from SocialMediaIE.models.helpers import AllenNLPSequential

logger = logging.getLogger(__name__)


def get_encoder(input_dim, output_dim, encoder_type, args):
    if encoder_type == "pass":
        return PassThroughEncoder(input_dim)
    if encoder_type == "bilstm":
        return PytorchSeq2SeqWrapper(
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
    if encoder_type == "self_attention":
        return IntraSentenceAttentionEncoder(
            input_dim=input_dim, projection_dim=output_dim
        )
    if encoder_type == "stacked_self_attention":
        return StackedSelfAttentionEncoder(
            input_dim=input_dim,
            hidden_dim=output_dim,
            projection_dim=output_dim,
            feedforward_hidden_dim=output_dim,
            num_attention_heads=5,
            num_layers=3,
            dropout_prob=args.dropout,
        )
    raise RuntimeError(f"Unknown encoder type={encoder_type}")


def get_task_encoder_dict(args, task_suffixes: Set[str], elmo_embedding_dim: int):
    hidden_dim = args.hidden_dim
    dropout = args.dropout
    encoder_type = args.encoder_type
    residual_connection = args.residual_connection
    def get_wrapped_encoder(encoder_list):
        return PytorchSeq2SeqWrapper(
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
                ["pos", "ner", "chunk", "ccg"]
            ), f"All tasks should be present. Found {task_suffixes}"
            logger.info(
                f"For multi_task_mode={multi_task_mode}, ignoring encoder_type={args.encoder_type}. All encoders are biLSTM"
            )
            INTERMEDIATE_INPUT = 2 * hidden_dim
            encoder_type = "bilstm-unwrapped"
            pos_encoder = get_encoder(elmo_embedding_dim, hidden_dim, encoder_type, args)
            chunk_encoder = get_encoder(INTERMEDIATE_INPUT, hidden_dim, encoder_type, args)
            ccg_encoder = get_encoder(INTERMEDIATE_INPUT, hidden_dim, encoder_type, args)
            ner_encoder = get_encoder(INTERMEDIATE_INPUT, hidden_dim, encoder_type, args)
            encoders = torch.nn.ModuleDict(
                {
                    "pos": get_wrapped_encoder([pos_encoder]),
                    "chunk": get_wrapped_encoder([pos_encoder, chunk_encoder]),
                    "ccg": get_wrapped_encoder([pos_encoder, chunk_encoder, ccg_encoder]),
                    "ner": get_wrapped_encoder([pos_encoder, chunk_encoder, ccg_encoder, ner_encoder]),
                }
            )
        elif multi_task_mode == "stacked-allennlp":
            DEFAULT_TASK_ORDER=["pos", "ner", "chunk", "ccg"]
            task_order = DEFAULT_TASK_ORDER
            if hasattr(args, "task_order"):
                task_order = args.task_order
            task_order = [k for k in task_order if k in task_suffixes]
            n_tasks = len(task_order)
        else:
            raise RuntimeError(f"multi_task_mode={multi_task_mode} not implemented.")
    return encoders


class MultiTaskCRFTagger(Model):
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
        self.crftagger = torch.nn.ModuleDict()
        self.metrics = dict()
        self._inference_mode = False
        for task in tasks:
            tag_namespace = task.tag_namespace
            self.crftagger[tag_namespace] = CrfTagger(
                vocab=vocab,
                text_field_embedder=self.word_embeddings,
                encoder=self.encoders[task.task_type],
                label_namespace=tag_namespace,
                label_encoding=task.label_encoding,
                calculate_span_f1=task.calculate_span_f1,
            )

    def set_inference_mode(self, mode=True):
        """Only compute the CRF for all tasks if inference mode is one. 
        Otherwise computing CRF for each task during training and evaluation is waste.
        CRF computation is N^2 operation and hence slows the training."""
        self._inference_mode = mode

    def get_task_tags(self, inputs, tag_namespace_indicies, tag_namespace):
        tag_namespace_idx = self.vocab.get_token_index(tag_namespace, "tag_namespace")
        tag_namespace_batch_indicies = (
            (tag_namespace_indicies == tag_namespace_idx).nonzero().squeeze(-1)
        )
        tags = None
        if tag_namespace_batch_indicies.shape[0] > 0:
            tags = inputs[tag_namespace][tag_namespace_batch_indicies]
        return tags

    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        tag_namespace_indicies = inputs["tag_namespace"]
        output = {}
        output["loss"] = 0.0
        for task in self._tasks:
            # Find valid task output indicies
            tag_namespace = task.tag_namespace
            tags = self.get_task_tags(inputs, tag_namespace_indicies, tag_namespace)
            if tags is None:
                crf_outputs = {}  # Set as empty dict
                if self._inference_mode:
                    crf_outputs = self.crftagger[tag_namespace].forward(
                        tags=tags, **inputs
                    )
                    output[f"{tag_namespace}_tags"] = crf_outputs["tags"]
            else:
                # Only compute CRF when tag is not none.
                crf_outputs = self.crftagger[tag_namespace].forward(tags=tags, **inputs)
                output[f"{tag_namespace}_tags"] = crf_outputs["tags"]

            if "loss" in crf_outputs and tags is not None:
                num_instances = tags.shape[0]
                output[f"{tag_namespace}_loss"] = crf_outputs["loss"] / num_instances
                output["loss"] += output[f"{tag_namespace}_loss"]
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for task in self._tasks:
            tag_namespace = task.tag_namespace
            crf_metrics = self.crftagger[tag_namespace].get_metrics(reset)
            for metric_name, metric_value in crf_metrics.items():
                metrics[f"{tag_namespace}_{metric_name}"] = metric_value
        return metrics
