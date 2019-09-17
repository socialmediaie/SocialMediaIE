import logging
from typing import Dict, Iterable, Iterator, List, Optional, Union, Set, Any


from allennlp.common import Params, Registrable
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.models.model import Model
from allennlp.data.iterators import BucketIterator, DataIterator
from allennlp.data.instance import Instance
from allennlp.training.trainer import TensorboardWriter


import torch
import tqdm
import os

from SocialMediaIE.data.multi_task_iterator import DatasetMingler

logger = logging.getLogger(__name__)


class Task(object):
    def __init__(
        self,
        tag_namespace: str,
        task_type: str,
        label_encoding: str = None,
        calculate_span_f1: bool = None,
        is_classification: bool = False
    ):
        self.tag_namespace = tag_namespace
        self.task_type = task_type
        self.label_encoding = label_encoding
        self.calculate_span_f1 = calculate_span_f1
        self.is_classification = is_classification

    def __repr__(self):
        return (
            f"Task(tag_namespace={self.tag_namespace!r}, "
            f"task_type={self.task_type}, "
            f"label_encoding={self.label_encoding}, "
            f"calculate_span_f1={self.calculate_span_f1}"
            f"is_classification={self.is_classification}"
            ")"
        )


class MultiTaskTrainer(Registrable):
    """
    A simple trainer that works in our multi-task setup.
    Really the main thing that makes this task not fit into our
    existing trainer is the multiple datasets.
    """

    def __init__(
        self,
        model: Model,
        serialization_dir: str,
        iterator: DataIterator,
        mingler: DatasetMingler,
        optimizer: torch.optim.Optimizer,
        train_datasets: Dict[str, Iterable[Instance]],
        validation_datasets: Dict[str, Iterable[Instance]],
        num_epochs: int = 10,
        num_serialized_models_to_keep: int = 10,
        cuda_device: int = -1,
    ) -> None:

        self.model = model
        self.iterator = iterator
        self.mingler = mingler
        self.optimizer = optimizer
        self.train_datasets = train_datasets
        self.validation_datasets = validation_datasets
        self.num_epochs = num_epochs
        self._serialization_dir = serialization_dir

        train_log = SummaryWriter(os.path.join(self._serialization_dir, "log", "train"))
        validation_log = SummaryWriter(
            os.path.join(self._serialization_dir, "log", "validation")
        )
        self._tensorboard = TensorboardWriter(
            train_log=train_log, validation_log=validation_log
        )

    def train(self) -> Dict:
        start_epoch = self.restore_checkpoint()

        self.model.train()
        for epoch in range(start_epoch, self.num_epochs):
            total_loss = 0.0
            batches = tqdm.tqdm(
                self.iterator(self.mingler.mingle(self.datasets), num_epochs=1)
            )
            for i, batch in enumerate(batches):
                self.optimizer.zero_grad()
                loss = self.model.forward(**batch)["loss"]
                loss.backward()
                total_loss += loss.item()
                self.optimizer.step()
                batches.set_description(f"epoch: {epoch} loss: {total_loss / (i + 1)}")

            # Save checkpoint
            self.save_checkpoint(epoch)

        return {}

    @classmethod
    def from_params(
        cls,  # type: ignore
        params: Params,
        serialization_dir: str,
        recover: bool = False,
    ) -> "MultiTaskTrainer":
        readers = {
            name: DatasetReader.from_params(reader_params)
            for name, reader_params in params.pop("train_dataset_readers").items()
        }
        train_file_paths = params.pop("train_file_paths").as_dict()

        datasets = {
            name: reader.read(train_file_paths[name])
            for name, reader in readers.items()
        }

        instances = (instance for dataset in datasets.values() for instance in dataset)
        vocab = Vocabulary.from_params(Params({}), instances)
        model = Model.from_params(params.pop("model"), vocab=vocab)
        iterator = DataIterator.from_params(params.pop("iterator"))
        iterator.index_with(vocab)
        mingler = DatasetMingler.from_params(params.pop("mingler"))

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

        num_epochs = params.pop_int("num_epochs", 10)

        _ = params.pop("trainer", Params({}))

        params.assert_empty(__name__)

        return MultiTaskTrainer(
            model, serialization_dir, iterator, mingler, optimizer, datasets, num_epochs
        )
