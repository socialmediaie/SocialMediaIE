from allennlp.data.dataset import Batch
from allennlp.nn.util import move_to_device

import torch

from collections import defaultdict

from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def evaluate_on_file(model, reader, iterator, filepath, metric_prefix, cuda_device=-1):
    torch.cuda.empty_cache()
    data = reader.read(filepath)
    model.get_metrics(reset=True)
    for batch in tqdm(iterator(data, num_epochs=1, shuffle=False), desc=f"{metric_prefix} - {filepath}"):
        if cuda_device >= 0:
            batch = move_to_device(batch, cuda_device)
        model(**batch)
    metrics = model.get_metrics(reset=True)
    metrics = {k: v for k,v in metrics.items() if k.startswith(metric_prefix)}
    return metrics


def evaluate_multiple_data(model, readers, iterator, filepaths, cuda_device=-1):
    all_metrics = defaultdict(dict)
    for tag_namespace, filepath_list in filepaths.items():
        reader = readers[tag_namespace]
        all_metrics[tag_namespace] = {}
        for filepath in filepath_list:
            metrics = evaluate_on_file(model, reader, iterator, filepath, tag_namespace, cuda_device)
            logger.info("\t".join([f"{k}: {v:.5f}" for k,v in metrics.items()]))
            all_metrics[tag_namespace][filepath] = metrics
    return all_metrics