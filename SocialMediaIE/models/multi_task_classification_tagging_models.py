from allennlp.models.model import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder

from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

from allennlp.modules.seq2seq_encoders import (
    PytorchSeq2SeqWrapper,
    StackedSelfAttentionEncoder,
    PassThroughEncoder,
    IntraSentenceAttentionEncoder,
)
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.crf_tagger import CrfTagger


import torch
import logging
from typing import Dict, Iterable, Iterator, List, Optional, Union, Set, Any

from SocialMediaIE.training.multitask_trainer import Task

from SocialMediaIE.models.helpers import AllenNLPSequential
from SocialMediaIE.models.multi_task_classification_models import BasicClassifierWithMetrics


TAGGING_TASKS = set(["pos", "ner", "chunk", "ccg"])
CLASSIFICATION_TASKS = set(["sentiment", "abusive", "uncertainity"])

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
    def get_wrapped_encoder(encoder_list, wrapper_class):
        # wrapper_class = PytorchSeq2VecWrapper for classification
        # wrapper_class = PytorchSeq2SeqWrapper for tagging
        return wrapper_class(
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
        raise NotImplementedError(f"multi_task_mode=single not implemented.")
        single_task_suffix = tuple(task_suffixes)[0]
        encoders = torch.nn.ModuleDict(
            {single_task_suffix: get_encoder(elmo_embedding_dim, hidden_dim, encoder_type, args)}
        )
    else:
        # More than one type of task
        multi_task_mode = args.multi_task_mode
        if multi_task_mode == "shared":
            #raise NotImplementedError(f"multi_task_mode={multi_task_mode} not implemented.")
            #shared_encoder = get_encoder(elmo_embedding_dim, hidden_dim, encoder_type, args)
            INTERMEDIATE_INPUT = 2 * hidden_dim
            encoder_type = "bilstm-unwrapped"
            base_shared_encoder = get_encoder(elmo_embedding_dim, hidden_dim, encoder_type, args)
            encoder_dict = {}
            ## Tagging encoders
            if len(set(task_suffixes) & set(TAGGING_TASKS)) > 0:
                shared_tagging_encoder = get_encoder(INTERMEDIATE_INPUT, hidden_dim, encoder_type, args)
                tagging_encoder_list = torch.nn.ModuleList([
                    base_shared_encoder, shared_tagging_encoder
                ])
                encoder_dict.update({
                    suffix: get_wrapped_encoder(
                        tagging_encoder_list,
                        PytorchSeq2SeqWrapper
                    )
                    for suffix in task_suffixes
                    if suffix in TAGGING_TASKS
                })

            ## Classification encoders
            if len(set(task_suffixes) & set(CLASSIFICATION_TASKS)) > 0:
                shared_classification_encoder = get_encoder(INTERMEDIATE_INPUT, hidden_dim, encoder_type, args)
                classification_encoder_list = torch.nn.ModuleList([
                    base_shared_encoder, shared_classification_encoder
                ])
                encoder_dict.update({
                    suffix: get_wrapped_encoder(
                        classification_encoder_list,
                        PytorchSeq2VecWrapper
                    )
                    for suffix in task_suffixes
                    if suffix in CLASSIFICATION_TASKS
                })
            encoders = torch.nn.ModuleDict(encoder_dict)
            #print(encoders)
        elif multi_task_mode == "invidual":
            raise NotImplementedError(f"multi_task_mode={multi_task_mode} not implemented.")
            # should be same as using full task models here for comparison
            encoders = torch.nn.ModuleDict(
                {
                    suffix: get_encoder(elmo_embedding_dim, hidden_dim, encoder_type, args)
                    for suffix in task_suffixes
                }
            )
        elif multi_task_mode == "stacked":
            assert task_suffixes == set(
                ["pos", "ner", "chunk", "ccg", "sentiment", "abusive", "uncertainity"]
            ), f"All tasks should be present. Found {task_suffixes}"
            logger.info(
                f"For multi_task_mode={multi_task_mode}, ignoring encoder_type={args.encoder_type}. All encoders are biLSTM"
            )
            INTERMEDIATE_INPUT = 2 * hidden_dim
            encoder_type = "bilstm-unwrapped"
            base_shared_encoder = get_encoder(elmo_embedding_dim, hidden_dim, encoder_type, args)

            ## Tagging encoders
            pos_encoder = get_encoder(INTERMEDIATE_INPUT, hidden_dim, encoder_type, args)
            chunk_encoder = get_encoder(INTERMEDIATE_INPUT, hidden_dim, encoder_type, args)
            ccg_encoder = get_encoder(INTERMEDIATE_INPUT, hidden_dim, encoder_type, args)
            ner_encoder = get_encoder(INTERMEDIATE_INPUT, hidden_dim, encoder_type, args)

            tagging_encoder_list = torch.nn.ModuleList([
                base_shared_encoder, pos_encoder, chunk_encoder, ccg_encoder, ner_encoder
            ])

            ## Classification encoders
            sentiment_encoder = get_encoder(INTERMEDIATE_INPUT, hidden_dim, encoder_type, args)
            abusive_encoder = get_encoder(INTERMEDIATE_INPUT, hidden_dim, encoder_type, args)
            uncertainity_encoder = get_encoder(INTERMEDIATE_INPUT, hidden_dim, encoder_type, args)

            classification_encoder_list = torch.nn.ModuleList([
                base_shared_encoder, sentiment_encoder, abusive_encoder, uncertainity_encoder
            ])
            encoders = torch.nn.ModuleDict(
                {
                    "pos": get_wrapped_encoder(tagging_encoder_list[:2], PytorchSeq2SeqWrapper),
                    "chunk": get_wrapped_encoder(tagging_encoder_list[:3], PytorchSeq2SeqWrapper),
                    "ccg": get_wrapped_encoder(tagging_encoder_list[:4], PytorchSeq2SeqWrapper),
                    "ner": get_wrapped_encoder(tagging_encoder_list[:5], PytorchSeq2SeqWrapper),
                    "sentiment": get_wrapped_encoder(classification_encoder_list[:2], PytorchSeq2VecWrapper),
                    "abusive": get_wrapped_encoder(classification_encoder_list[:3], PytorchSeq2VecWrapper),
                    "uncertainity": get_wrapped_encoder(classification_encoder_list[:4], PytorchSeq2VecWrapper),
                }
            )
        elif multi_task_mode == "stacked-allennlp":
            raise NotImplementedError(f"multi_task_mode={multi_task_mode} not implemented.")
            DEFAULT_TASK_ORDER=["pos", "ner", "chunk", "ccg"]
            task_order = DEFAULT_TASK_ORDER
            if hasattr(args, "task_order"):
                task_order = args.task_order
            task_order = [k for k in task_order if k in task_suffixes]
            n_tasks = len(task_order)
        else:
            raise NotImplementedError(f"multi_task_mode={multi_task_mode} not implemented.")
    return encoders

class MultiTaskCRFTaggerAndClassifier(Model):
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
        self.tagger_or_classifier = torch.nn.ModuleDict()
        self.metrics = dict()
        self._inference_mode = False
        for task in tasks:
            tag_namespace = task.tag_namespace
            if task.task_type in TAGGING_TASKS:
                self.tagger_or_classifier[tag_namespace] = CrfTagger(
                    vocab=vocab,
                    text_field_embedder=self.word_embeddings,
                    encoder=self.encoders[task.task_type],
                    label_namespace=tag_namespace,
                    label_encoding=task.label_encoding,
                    calculate_span_f1=task.calculate_span_f1,
                )
            elif task.task_type in CLASSIFICATION_TASKS:
                self.tagger_or_classifier[tag_namespace] = BasicClassifierWithMetrics(
                    vocab=vocab,
                    text_field_embedder=self.word_embeddings,
                    seq2vec_encoder=self.encoders[task.task_type],
                    label_namespace=tag_namespace,
                )
            else:
                raise NotImplementedError(f"model for task.task_type={task.task_type} not implemented.")

    def set_inference_mode(self, mode=True):
        """Only compute the CRF for all tasks if inference mode is one. 
        Otherwise computing CRF for each task during training and evaluation is waste.
        CRF computation is N^2 operation and hence slows the training."""
        self._inference_mode = mode

    def get_task_tags(self, inputs, tag_namespace_indicies, tag_namespace, namespace_vocab_key="tag_namespace"):
        tag_namespace_idx = self.vocab.get_token_index(tag_namespace, namespace_vocab_key)
        tag_namespace_batch_indicies = (
            (tag_namespace_indicies == tag_namespace_idx).nonzero().squeeze(-1)
        )
        tags = None
        if tag_namespace_batch_indicies.shape[0] > 0:
            tags = inputs[tag_namespace][tag_namespace_batch_indicies]
        return tags

    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        output = {}
        output["loss"] = 0.0
        for task in self._tasks:
            # Find valid task output indicies
            tag_namespace = task.tag_namespace
            output_key = "tags"
            tag_namespace_indicies = None
            namespace_vocab_key = "tag_namespace"
            forward_outputs_dict = dict(tags=None, label=None)
            forward_output_name = "tags"
            if task.task_type in TAGGING_TASKS:
                output_key = "tags"                
                namespace_vocab_key = "tag_namespace"
                forward_output_name = "tags"
            elif task.task_type in CLASSIFICATION_TASKS:
                output_key = "probs"
                namespace_vocab_key = "label_namespace"
                forward_output_name = "label"
            else:
                raise NotImplementedError(f"task={task} not implemented for tag_type={task.task_type}")
            tag_namespace_indicies = inputs.get(namespace_vocab_key) # This will always be non None
            tags = None
            if tag_namespace_indicies is not None:
                tags = self.get_task_tags(inputs, tag_namespace_indicies, tag_namespace, namespace_vocab_key)
            forward_outputs_dict[forward_output_name] = tags
            if tags is None:
                tagger_or_classifier_outputs = {}  # Set as empty dict
                if self._inference_mode:
                    tagger_or_classifier_outputs = self.tagger_or_classifier[tag_namespace].forward(
                        **forward_outputs_dict, **inputs
                    )
                    output[f"{tag_namespace}_{output_key}"] = tagger_or_classifier_outputs[output_key]
            else:
                # Only compute CRF when tag is not none.
                tagger_or_classifier_outputs = self.tagger_or_classifier[tag_namespace].forward(**forward_outputs_dict, **inputs)
                output[f"{tag_namespace}_{output_key}"] = tagger_or_classifier_outputs[output_key]

            if "loss" in tagger_or_classifier_outputs and tags is not None:
                num_instances = tags.shape[0]
                output[f"{tag_namespace}_loss"] = tagger_or_classifier_outputs["loss"] / num_instances
                output["loss"] += output[f"{tag_namespace}_loss"]
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for task in self._tasks:
            tag_namespace = task.tag_namespace
            tagger_or_classifier_metrics = self.tagger_or_classifier[tag_namespace].get_metrics(reset)
            for metric_name, metric_value in tagger_or_classifier_metrics.items():
                metrics[f"{tag_namespace}_{metric_name}"] = metric_value
        return metrics
