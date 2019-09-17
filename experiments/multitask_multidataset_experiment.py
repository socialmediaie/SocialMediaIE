# coding: utf-8


import argparse
import json
import logging
import os
import shutil
from io import StringIO
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

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
    AllenNLPSequential,
    MultiTaskCRFTagger,
    get_task_encoder_dict,
)
from SocialMediaIE.training.multitask_trainer import Task

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.ERROR,
)
logger = logging.getLogger(__name__)

TASK_CONFIGS = {}
# Add POS tasks
TASK_CONFIGS.update(
    {k: Task(k, task_type="pos") for k in ["ptb_pos", "ark_pos", "ud_pos"]}
)
# Add CHUNK tasks
TASK_CONFIGS.update(
    {
        k: Task(k, task_type="chunk", label_encoding="BIO", calculate_span_f1=True)
        for k in ["ritter_chunk"]
    }
)
# Add NER tasks
TASK_CONFIGS.update(
    {
        k: Task(k, task_type="ner", label_encoding="BIO", calculate_span_f1=True)
        for k in [
            "yodie_ner",
            "ritter_ner",
            "wnut17_ner",
            "neel_ner",
            "broad_ner",
            "multimodal_ner",
        ]
    }
)
# Add CCG tasks
TASK_CONFIGS.update(
    {
        k: Task(k, task_type="ccg", label_encoding="BIO", calculate_span_f1=True)
        for k in ["ritter_ccg"]
    }
)

# POS_TASKS = [Task(k, task_type="pos") for k in ["ptb_pos", "ark_pos", "ud_pos"]]
# CHUNK_TASKS = [Task(k, task_type="chunk", label_encoding="BIO", calculate_span_f1=True) for k in ["ritter_chunk"]]
# NER_TASKS = [Task(k, task_type="ner", label_encoding="BIO", calculate_span_f1=True)
#             for k in ["yodie_ner", "ritter_ner", "wnut17_ner", "neel_ner", "broad_ner", "multimodal_ner"]]
# TASKS = POS_TASKS + CHUNK_TASKS + NER_TASKS
# options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
# weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# Compute two different representation for each token.
# Each representation is a linear weighted combination for the
# 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
# elmo = Elmo(options_file, weight_file, 2, dropout=0)

# use batch_to_ids to convert sentences to character ids
# sentences = [['First', 'sentence', '.'], ['Another', '.']]
# character_ids = batch_to_ids(sentences)


"""token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), 
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
"""
# Use the 'Small' pre-trained model
# options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
#                 '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json')
# weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
#                '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')

options_file = (
    "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/"
    "2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
)
weight_file = (
    "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/"
    "2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
)

# pos_encoder = torch.nn.LSTM(ELMO_EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=BIDIRECTIONAL, dropout=DROPOUT)
# chunk_encoder = torch.nn.LSTM(INTERMEDIATE_INPUT, HIDDEN_DIM, batch_first=True, bidirectional=BIDIRECTIONAL, dropout=DROPOUT)
# ner_encoder = torch.nn.LSTM(INTERMEDIATE_INPUT, HIDDEN_DIM, batch_first=True, bidirectional=BIDIRECTIONAL, dropout=DROPOUT)
# logger.info(f"combined_training_dataset={len(combined_training_dataset)}")
# logger.info(f"combined_validation_dataset={len(combined_validation_dataset)}")


# iterator = BucketIterator(batch_size=10, sorting_keys=[("tokens", "num_tokens")])


def get_parser():
    parser = argparse.ArgumentParser(description="MultiTask model")

    data_spec = parser.add_argument_group("data_spec")
    data_spec.add_argument(
        "--task",
        nargs="+",
        type=str,
        help="Task namespaces should be part of dataset_paths",
    )
    data_spec.add_argument(
        "--dataset-paths-file",
        required=True,
        type=str,
        help="JSON file with all dataset paths.",
    )
    data_spec.add_argument(
        "--dataset-path-prefix",
        default="",
        type=str,
        help="Path prefix to be attached to each path.",
    )
    data_spec.add_argument(
        "--model-dir",
        default="../data/models/websci_mt/",
        type=str,
        help="Directory where the model will be saved",
    )
    data_spec.add_argument(
        "--clean-model-dir", action="store_true", help="Force delete model directory"
    )

    model_spec = parser.add_argument_group("model_spec")
    model_spec.add_argument(
        "--proj-dim", default=100, type=int, help="Projected dim for elmo embedding"
    )
    model_spec.add_argument(
        "--hidden-dim", default=100, type=int, help="Hidden dim for model"
    )
    model_spec.add_argument(
        "--encoder-type",
        default="pass",
        type=str,
        choices=["pass", "bilstm", "self_attention", "stacked_self_attention"],
        help="Possible encoder types",
    )
    model_spec.add_argument(
        "--multi-task-mode",
        default="shared",
        type=str,
        choices=["shared", "invidual", "stacked"],
        help="Multi task modes.",
    )
    model_spec.add_argument(
        "--dropout", default=0.5, type=float, help="Dropout across layers"
    )
    model_spec.add_argument(
        "--residual-connection", action="store_true", help="Use residual connection"
    )

    training_spec = parser.add_argument_group("training_spec")
    training_spec.add_argument(
        "--lr", default=1e-2, type=float, help="Learning rate for optimizer"
    )
    training_spec.add_argument(
        "--weight-decay", default=0, type=float, help="Weight decay"
    )
    training_spec.add_argument("--batch-size", default=16, type=int, help="Batch size")
    training_spec.add_argument(
        "--epochs", default=10, type=int, help="Number of epochs"
    )
    training_spec.add_argument(
        "--patience",
        default=3,
        type=int,
        help="Number of epochs before stopping training without decrease in validation loss",
    )
    training_spec.add_argument("--cuda", action="store_true", help="Use CUDA device.")
    training_spec.add_argument(
        "--test-mode", action="store_true", help="Only test using an existing model."
    )

    return parser


def cuda_device(has_cuda):
    device_id = -1
    if has_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(f"No cuda available. Remove flag --cuda.")
        else:
            device_id = 0
            logger.info(f"Found cuda. Using cuda:{device_id}")
    return device_id  # id of CUDA GPU


def get_all_dataset_paths(json_file, path_prefix=""):
    with open(json_file, encoding="utf-8") as fp:
        path_json = json.load(fp)
    output_json = {}
    for task, splits in path_json.items():
        output_json[task] = {}
        for split_key, paths in splits.items():
            output_json[task][split_key] = {}
            updated_paths = []
            for path in paths:
                path = os.path.realpath(os.path.join(path_prefix, path))
                updated_paths.append(path)
            output_json[task][split_key] = updated_paths
    return output_json


def main(args):
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

    if not TEST_MODE:
        train_dataset = read_datasets(dataset_paths, readers, data_split="train")
        validation_dataset = read_datasets(dataset_paths, readers, data_split="dev")

        vocab = create_vocab([train_dataset, validation_dataset])

        # Special case for CCG
        if "ccg" in task_suffixes or "pos" in task_suffixes:
            for task in TASKS:
                if task.task_type == "ccg":
                    for tag in ["B-NOUN.SHAPE", "I-NOUN.PROCESS"]:
                        vocab.add_token_to_namespace(tag, task.tag_namespace)
                if task.tag_namespace == "ud_pos":
                    for tag in ["CONJ"]:
                        vocab.add_token_to_namespace(tag, task.tag_namespace)
				
    else:
        vocab = Vocabulary.from_files(os.path.join(SERIALIZATION_DIR, "vocabulary"))

    # encoder = PassThroughEncoder(ELMO_EMBEDDING_DIM)
    model = MultiTaskCRFTagger(word_embeddings, encoders, vocab, TASKS)
    model = model.cuda(device=CUDA_DEVICE)

    if not TEST_MODE:
        iterator = CustomHomogeneousBatchIterator(
            partition_key="dataset", batch_size=BATCH_SIZE, cache_instances=True
        )
        iterator.index_with(vocab)

        if CLEAN_MODEL_DIR:
            if os.path.exists(SERIALIZATION_DIR):
                logger.info(f"Deleting {SERIALIZATION_DIR}")
                shutil.rmtree(SERIALIZATION_DIR)
            logger.info(f"Creating {SERIALIZATION_DIR}")
            os.makedirs(SERIALIZATION_DIR)

        logger.info(f"Writing arguments to arguments.json in {SERIALIZATION_DIR}")
        with open(os.path.join(SERIALIZATION_DIR, "arguments.json"), "w+") as fp:
            json.dump(vars(args), fp, indent=2)

        logger.info(f"Writing vocabulary in {SERIALIZATION_DIR}")
        vocab.save_to_files(os.path.join(SERIALIZATION_DIR, "vocabulary"))
        # Use list to ensure each epoch is a full pass through the data
        combined_training_dataset = list(roundrobin_iterator(*train_dataset.values()))
        combined_validation_dataset = list(
            roundrobin_iterator(*validation_dataset.values())
        )

        # optimizer = optim.ASGD(model.parameters(), lr=0.01, t0=100, weight_decay=0.1)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        training_stats = []
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            iterator=iterator,
            train_dataset=combined_training_dataset,
            validation_dataset=combined_validation_dataset,
            patience=PATIENTCE,
            num_epochs=NUM_EPOCHS,
            cuda_device=CUDA_DEVICE,
            serialization_dir=SERIALIZATION_DIR,
            # model_save_interval=600
        )
        stats = trainer.train()
        training_stats.append(stats)

        with open(os.path.join(SERIALIZATION_DIR, "training_stats.json"), "w+") as fp:
            json.dump(training_stats, fp, indent=2)
    else:
        model.load_state_dict(torch.load(os.path.join(SERIALIZATION_DIR, "best.th")))
        model = model.cuda(device=CUDA_DEVICE)

    # Empty cache to ensure larger batch can be loaded for testing
    torch.cuda.empty_cache()

    test_filepaths = {
        task.tag_namespace: dataset_paths[task.tag_namespace]["test"] for task in TASKS
    }

    logger.info("Evaluating on test data")

    test_iterator = CustomHomogeneousBatchIterator(
        partition_key="dataset", batch_size=BATCH_SIZE * 2
    )
    test_iterator.index_with(vocab)
    model = model.eval()
    test_stats = evaluate_multiple_data(
        model, readers, test_iterator, test_filepaths, cuda_device=CUDA_DEVICE
    )
    with open(os.path.join(SERIALIZATION_DIR, "test_stats.json"), "w+") as fp:
        json.dump(test_stats, fp, indent=2)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger.info(args)
    main(args)
