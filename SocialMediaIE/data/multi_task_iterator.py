from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Union

from allennlp.common import Params, Registrable
from allennlp.common.util import lazy_groups_of
from allennlp.common.file_utils import cached_path
from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.data.fields import MetadataField
from allennlp.data.iterators import DataIterator
from allennlp.data.vocabulary import Vocabulary

from itertools import cycle, islice

import random


class DatasetMingler(Registrable):
    """
    Our ``DataIterator`` class expects a single dataset;
    this is an abstract class for combining multiple datasets into one.
    You could imagine an alternate design where there is a
    ``MinglingDatasetReader`` that wraps multiple dataset readers,
    but then somehow you'd have to get it multiple file paths.
    """

    def mingle(self, datasets: Dict[str, Iterable[Instance]]) -> Iterable[Instance]:
        raise NotImplementedError


@DatasetMingler.register("round-robin")
class RoundRobinMingler(DatasetMingler):
    """
    Cycle through datasets, ``take_at_time`` instances at a time.
    """

    def __init__(
        self, dataset_name_field: str = "dataset", take_at_a_time: int = 1
    ) -> None:
        self.dataset_name_field = dataset_name_field
        self.take_at_a_time = take_at_a_time

    def mingle(self, datasets: Dict[str, Iterable[Instance]]) -> Iterable[Instance]:
        iterators = {name: iter(dataset) for name, dataset in datasets.items()}
        done: Set[str] = set()

        while iterators.keys() != done:
            for name, iterator in iterators.items():
                if name not in done:
                    try:
                        for _ in range(self.take_at_a_time):
                            instance = next(iterator)
                            instance.fields[self.dataset_name_field] = MetadataField(
                                name
                            )
                            yield instance
                    except StopIteration:
                        done.add(name)


@DataIterator.register("custom_homogeneous_batch")
class CustomHomogeneousBatchIterator(DataIterator):
    """
    This iterator takes a dataset of potentially heterogeneous instances
    and yields back homogeneous batches. It assumes that each instance has
    some ``MetadataField`` indicating what "type" of instance it is
    and bases its notion of homogeneity on that (and, in particular, not on
    inspecting the "field signature" of the instance.)

    Parameters
    ----------
    batch_size : ``int``, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    instances_per_epoch : ``int``, optional, (default = None)
        If specified, each epoch will consist of precisely this many instances.
        If not specified, each epoch will consist of a single pass through the dataset.
    max_instances_in_memory : ``int``, optional, (default = None)
        If specified, the iterator will load this many instances at a time into an
        in-memory list and then produce batches from one such list at a time. This
        could be useful if your instances are read lazily from disk.
    cache_instances : ``bool``, optional, (default = False)
        If true, the iterator will cache the tensorized instances in memory.
        If false, it will do the tensorization anew each iteration.
    track_epoch : ``bool``, optional, (default = False)
        If true, each instance will get a ``MetadataField`` containing the epoch number.
    partition_key : ``str``, optional, (default = "dataset")
        The key of the ``MetadataField`` indicating what "type" of instance this is.
    """
    def __init__(self,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 partition_key: str = "dataset") -> None:
        super().__init__(batch_size, instances_per_epoch, max_instances_in_memory,
                         cache_instances, track_epoch)
        self._partition_key = partition_key

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # First break the dataset into memory-sized lists:
        for instance_list in self._memory_sized_lists(instances):
            if shuffle:
                random.shuffle(instance_list)

            # Divvy up the instances based on their value of the "partition_key" field.
            hoppers: Dict[str, List[Instance]] = defaultdict(list)
            for instance in instance_list:
                partition = instance.fields[self._partition_key].metadata  # type: ignore
                hoppers[partition].append(instance)

            # Get a `lazy_groups_of` iterator over each set of homogeneous instances.
            batches = {key: lazy_groups_of(iter(hopper), self._batch_size) for key, hopper in hoppers.items()}

            remaining = set(batches)

            # Yield batches in a round-robin fashion until none are left.
            while remaining:
                # TODO: shuffle keys before each batch creation.
                # Another approach can be to sample a task proportional to its data probability.
                # Then sample a batch from that task. 
                # Data prob can be updated once the batch is sampled. 
                for key, lazy_batches in batches.items():
                    if key in remaining:
                        try:
                            batch = next(lazy_batches)
                            yield Batch(batch)
                        except StopIteration:
                            remaining.remove(key)



def roundrobin_iterator(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))