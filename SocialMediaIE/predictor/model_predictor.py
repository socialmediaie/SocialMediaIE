# coding: utf-8


import argparse
import json
import logging
import os
import shutil
from io import StringIO
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union
import pandas as pd



import torch
from allennlp.common import Params, Registrable
from allennlp.common.file_utils import cached_path

# for dataset reader
from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import (
    LabelField,
    MetadataField,
    SequenceLabelField,
    TextField,
)
from allennlp.data.iterators import BucketIterator, DataIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.models.archival import archive_model
from allennlp.models.crf_tagger import CrfTagger
from allennlp.modules import FeedForward
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules.seq2seq_encoders import (
    IntraSentenceAttentionEncoder,
    PassThroughEncoder,
    PytorchSeq2SeqWrapper,
    Seq2SeqEncoder,
    StackedSelfAttentionEncoder,
)
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, Seq2VecEncoder
from allennlp.modules.text_field_embedders import (
    BasicTextFieldEmbedder,
    TextFieldEmbedder,
)
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from allennlp.nn.util import move_to_device

# read pretrained embedding from AWS S3
from allennlp.modules.token_embedders.embedding import _read_embeddings_from_text_file
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import Trainer
from torch import optim

from SocialMediaIE.data.conll_data_reader import ConLLDatasetReader
from SocialMediaIE.data.multi_task_iterator import (
    CustomHomogeneousBatchIterator,
    RoundRobinMingler,
    roundrobin_iterator,
)
from SocialMediaIE.data.multi_task_reader import create_vocab, read_datasets
from SocialMediaIE.evaluation.evaluate_model import evaluate_multiple_data
from SocialMediaIE.models.multi_task_models import (
    MultiTaskCRFTagger,
    get_task_encoder_dict,
)
from SocialMediaIE.models.helpers import AllenNLPSequential

from SocialMediaIE.training.multitask_trainer import Task
from SocialMediaIE.scripts.multitask_multidataset_experiment import (
    TASK_CONFIGS, 
    options_file, 
    weight_file,
    get_parser,
    get_all_dataset_paths,
    cuda_device
)


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.ERROR,
)
logger = logging.getLogger(__name__)


def run(args):
    ALL_DATASET_PATHS = get_all_dataset_paths(
        args.dataset_paths_file, 
        args.dataset_path_prefix
    )
    SELECTED_TASK_NAMES = args.task
    PROJECTION_DIM = args.proj_dim
    HIDDEN_DIM = args.hidden_dim
    # BIDIRECTIONAL=True
    # INTERMEDIATE_INPUT=2*HIDDEN_DIM if BIDIRECTIONAL else HIDDEN_DIM
    DROPOUT = args.dropout
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    PATIENTCE = args.patience
    SERIALIZATION_DIR = args.model_dir
    CLEAN_MODEL_DIR = args.clean_model_dir
    CUDA_DEVICE = cuda_device(args.cuda)
    TEST_MODE = args.test_mode
    # device = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() and args.cuda else "cpu")

    TASKS = [TASK_CONFIGS[task_name] for task_name in SELECTED_TASK_NAMES]
    dataset_paths = {
        task_name: ALL_DATASET_PATHS[task_name] for task_name in SELECTED_TASK_NAMES
    }

    tag_namespace_hashing_fn = {
        tag_namespace: i for i, tag_namespace in enumerate(TASK_CONFIGS.keys())
    }.get

    elmo_token_indexer = ELMoTokenCharactersIndexer()
    token_indexers = {"tokens": elmo_token_indexer}
    readers = {
        task.tag_namespace: ConLLDatasetReader(
            task.tag_namespace,
            token_indexers=token_indexers,
            tag_namespace_hashing_fn=tag_namespace_hashing_fn,
        )
        for task in TASKS
    }

    elmo_embedder = ElmoTokenEmbedder(
        options_file,
        weight_file,
        requires_grad=False,
        dropout=DROPOUT,
        projection_dim=PROJECTION_DIM,
    )
    # elmo_embedder = Elmo(options_file, weight_file, num_output_representations=3)

    # Pass in the ElmoTokenEmbedder instance instead
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
    ELMO_EMBEDDING_DIM = elmo_embedder.get_output_dim()

    # POS -> CHUNK -> NER
    task_suffixes = set(
        [task_name.rsplit("_", 1)[-1] for task_name in SELECTED_TASK_NAMES]
    )
    encoders = get_task_encoder_dict(args, task_suffixes, ELMO_EMBEDDING_DIM)

    vocab = Vocabulary.from_files(os.path.join(SERIALIZATION_DIR, "vocabulary"))

    # encoder = PassThroughEncoder(ELMO_EMBEDDING_DIM)
    model = MultiTaskCRFTagger(word_embeddings, encoders, vocab, TASKS)
    model.load_state_dict(torch.load(os.path.join(SERIALIZATION_DIR, "best.th")))
    if args.cuda:
        model = model.cuda(device=CUDA_DEVICE)
        # Empty cache to ensure larger batch can be loaded for testing
        torch.cuda.empty_cache()
    
    logger.info("Evaluating on test data")

    test_iterator = CustomHomogeneousBatchIterator(
        partition_key="dataset", batch_size=BATCH_SIZE * 2
    )
    test_iterator.index_with(vocab)
    model = model.eval()
    model.set_inference_mode(True)
    return TASKS, vocab, model, readers, test_iterator


from copy import deepcopy
from collections import namedtuple

ARG_KEYS = ('task',
 'dataset_paths_file',
 'dataset_path_prefix',
 'model_dir',
 'clean_model_dir',
 'proj_dim',
 'hidden_dim',
 'encoder_type',
 'multi_task_mode',
 'dropout',
 'lr',
 'weight_decay',
 'batch_size',
 'epochs',
 'patience',
 'cuda',
 'test_mode',
 'residual_connection'
 )
PREFIX = os.path.realpath("../")

Arguments = namedtuple("ModelArgument", ARG_KEYS)

def get_args(prefix, serialization_dir):
    path = os.path.join(serialization_dir, "arguments.json")
    with open(path) as fp:
        args = json.load(fp)
    args = deepcopy(args)
    args["dataset_paths_file"] = os.path.realpath(os.path.join(prefix, *args["dataset_paths_file"].split("/")[-2:]))
    args["dataset_path_prefix"] = os.path.realpath(os.path.join(prefix, args["dataset_path_prefix"].split("/")[-1]))
    args["model_dir"] = os.path.realpath(serialization_dir)
    args["test_mode"] = True
    args["residual_connection"] = args.get("residual_connection", False)
    args = Arguments(*[args[k] for k in Arguments._fields])
    return args


def get_instance(tokens, reader, vocab):
    n = len(tokens)
    tag_namespace = reader._tag_namespace
    tags = [vocab.get_token_from_index(0, tag_namespace)]*n
    return reader.text_to_instance(tokens, tags)

def get_model_output(model, data, args, readers, vocab, test_iterator):
    # Get first reader
    reader = next(iter(readers.values()))
    data = (get_instance(tokens, reader, vocab) for tokens in data)
    with torch.no_grad():
        for batch in test_iterator(data, num_epochs=1, shuffle=False):
            if args.cuda:
                batch = move_to_device(batch, 0)
            output = model(**batch)
            valid_keys = [k for k in output.keys() if not k.endswith("loss")]
            for instance_outputs in zip(*[output[k] for k in valid_keys]):
                yield {k: v for k,v in zip(valid_keys, instance_outputs)}


def get_output_tokens(tags, vocab, key):
    return [vocab.get_token_from_index(t, key) for t in tags]


def output_to_df(tokens, output, vocab):
    seq_length = len(tokens)
    parsed_tags = {"tokens": tokens}
    for k, tags in output.items():
        if k.endswith("_tags"):
            k = k[:-5]
        else:
            continue
        parsed_tags[k] = get_output_tokens(tags, vocab, k)[:seq_length]
    return pd.DataFrame(parsed_tags)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger.info(args)
    run(args)
