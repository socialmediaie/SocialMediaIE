from typing import Iterator, List, Dict, Optional, Iterable, Union, Any

from allennlp.data.iterators import DataIterator
from allennlp.common.file_utils import cached_path
from allennlp.data.vocabulary import Vocabulary
from allennlp.common import Params
from itertools import chain

def read_datasets(
    path_dict:Dict[str, Dict[str, List[str]]], 
    readers:Dict[str, DataIterator], 
    data_split:str):
    """Create a large dictionary of datasets from a list of paths and readers"""
    datasets = {}
    for tag_namespace, splits in path_dict.items():
        for filepath in splits[data_split]:
            read_dataset = readers[tag_namespace].read(cached_path(filepath))
            if tag_namespace not in datasets:
                datasets[tag_namespace] = read_dataset
            else:
                datasets[tag_namespace] = chain(datasets[tag_namespace], read_dataset)
    return datasets


def create_vocab(list_of_dataset_dicts:Dict[str, Dict[str, List[str]]]):
    """Create a combined vocab from all the datasets"""
    non_padded_namespaces = ("tag_namespace", "*pos", "*chunk", "*ner", "*ccg")
    vocab = Vocabulary(non_padded_namespaces=non_padded_namespaces)
    for dataset_dict in list_of_dataset_dicts:
        for dataset in dataset_dict.values():
            params = Params({"non_padded_namespaces": non_padded_namespaces})
            vocab.extend_from_instances(params, dataset)
    return vocab

def create_classification_vocab(list_of_dataset_dicts:Dict[str, Dict[str, List[str]]]):
    """Create a combined vocab from all the datasets"""
    non_padded_namespaces = ("label_namespace", "*sentiment", "*abusive", "*uncertainity")
    vocab = Vocabulary(non_padded_namespaces=non_padded_namespaces)
    for dataset_dict in list_of_dataset_dicts:
        for dataset in dataset_dict.values():
            params = Params({"non_padded_namespaces": non_padded_namespaces})
            vocab.extend_from_instances(params, dataset)
    return vocab

def create_classification_tagging_vocab(list_of_dataset_dicts:Dict[str, Dict[str, List[str]]]):
    """Create a combined vocab from all the datasets"""
    non_padded_namespaces = (
        "tag_namespace", "*pos", "*chunk", "*ner", "*ccg", 
        "label_namespace", "*sentiment", "*abusive", "*uncertainity"
    )
    vocab = Vocabulary(non_padded_namespaces=non_padded_namespaces)
    for dataset_dict in list_of_dataset_dicts:
        for dataset in dataset_dict.values():
            params = Params({"non_padded_namespaces": non_padded_namespaces})
            vocab.extend_from_instances(params, dataset)
    return vocab
