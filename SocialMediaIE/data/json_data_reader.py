from typing import Dict, Iterable, Iterator, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from allennlp.common import Params, Registrable
from allennlp.common.file_utils import cached_path

# for dataset reader
from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import (
    Field,
    LabelField,
    MetadataField,
    SequenceLabelField,
    TextField,
)
from allennlp.data.iterators import BucketIterator, DataIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.vocabulary import Vocabulary

# for building model
from allennlp.models import Model
from allennlp.models.crf_tagger import CrfTagger
from allennlp.modules import FeedForward
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, Seq2VecEncoder
from allennlp.modules.text_field_embedders import (
    BasicTextFieldEmbedder,
    TextFieldEmbedder,
)
from allennlp.modules.token_embedders import Embedding

# read pretrained embedding from AWS S3
from allennlp.modules.token_embedders.embedding import _read_embeddings_from_text_file
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import Trainer


import json
from SocialMediaIE.data.tokenization import tokenize as tokenize_tweet

def read_json_file(fp):
    """Yield lines for each sequence from a conll style file. 
    Sequences are seperated by blank lines. 
    Each sequence can have multiple features in tab seperated columns."""
    for line in fp:
        line = line.rstrip()
        yield json.loads(line)
        


class JSONDatasetReader(DatasetReader):
    """
    DatasetReader for JSON style data for different tag_namespaces
    """

    def __init__(
        self,
        tag_namespace: str,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
        tag_namespace_hashing_fn = lambda x: x
    ) -> None:
        super().__init__(lazy)
        self._tag_namespace = tag_namespace
        self._tokenizer = tokenizer or tokenize_tweet
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tag_namespace_hashing_fn = tag_namespace_hashing_fn

    def _read(self, file_path: str) -> Iterator[Instance]:
        """Yield an instance with tokens and tags"""
        with open(cached_path(file_path), "r", encoding="utf-8") as data_file:
            for json_data in read_json_file(data_file):
                tokens = self._tokenizer(json_data["text"])
                label = json_data["label"]
                yield self.text_to_instance(tokens, label)

    def text_to_instance(self, tokens: List[Token], label: str) -> Instance:
        """Each instance has {"tokens": , "tag_namespace": , ${tag_namespace}: }"""
        # TextField requires ``Token`` objects
        tag_namespace_hashing_fn = self._tag_namespace_hashing_fn
        tokens = [Token(token) for token in tokens]
        sequence = TextField(tokens, self._token_indexers)
        tag_namespace = LabelField(self._tag_namespace, label_namespace="label_namespace")
        label = LabelField(label, label_namespace=self._tag_namespace)
        instance_fields: Dict[str, Field] = {
            "tokens": sequence,
            "label_namespace": tag_namespace,
            self._tag_namespace: label
        }
        # Set metadata using id of the tag_namespace string to reduce instance size in memory
        instance_fields["dataset"] = MetadataField(tag_namespace_hashing_fn(self._tag_namespace))
        """instance_fields['metadata'] = MetadataField({
            "tokens": tokens,
            "tags": tags
        })"""
        return Instance(instance_fields)

