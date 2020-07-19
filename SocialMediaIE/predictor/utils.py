from copy import deepcopy
from collections import namedtuple
import os
import json

ARG_KEYS = (
    'task',
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

def get_args(prefix, serialization_dir, cuda=None):
    path = os.path.join(serialization_dir, "arguments.json")
    with open(path) as fp:
        args = json.load(fp)
    args = deepcopy(args)
    args["dataset_paths_file"] = os.path.realpath(os.path.join(prefix, *args["dataset_paths_file"].split("/")[-2:]))
    args["dataset_path_prefix"] = os.path.realpath(os.path.join(prefix, args["dataset_path_prefix"].split("/")[-1]))
    args["model_dir"] = os.path.realpath(serialization_dir)
    args["test_mode"] = True
    args["residual_connection"] = args.get("residual_connection", False)
    if cuda:
        args["cuda"] = cuda
    args = Arguments(*[args[k] for k in Arguments._fields])
    return args


def split_tag(tag):
    return tuple(tag.split("-", 1)) if tag != "O" else (tag, None)


def extract_entities(tags):
    tags = list(tags)
    curr_entity = []
    entities = []
    for i, tag in enumerate(tags + ["O"]):
        # Add dummy tag in end to ensure the last entity is added to entities
        boundary, label = split_tag(tag)
        if curr_entity:
            # Exit entity
            if boundary in {"B", "O"} or label != curr_entity[-1][1]:
                start = i - len(curr_entity)
                end = i
                entity_label = curr_entity[-1][1]
                entities.append((entity_label, start, end))
                curr_entity = []
            elif boundary == "I":
                curr_entity.append((boundary, label))
        if boundary == "B":
            # Enter or inside entity
            assert not curr_entity, f"Entity should be empty. Found: {curr_entity}"
            curr_entity.append((boundary, label))
    return entities


def get_entity_info(bio_labels, tokens, text=None, spans=None):
    entities_info = extract_entities(bio_labels)
    entities = []
    for label, start, end in entities_info:
        entity_phrase = None
        if text and spans:
            start_char_idx = spans[start][0]
            end_char_idx = spans[end-1][1]
            entity_phrase = text[start_char_idx:end_char_idx]
        entities.append(dict(
            tokens=tokens[start:end],
            label=label,
            start=start,
            end=end,
            entity_phrase=entity_phrase))
    return entities


def get_df_entities(df, text=None):
    span_columns = [
        c for c in df.columns if c.endswith(("_ner", "_chunk", "_ccg"))
    ]
    tokens = list(df["tokens"])
    spans = list(df["span"])
    task_entities = {c: [] for c in span_columns}
    for c in span_columns:
        bio_labels = df[c]
        task_entities[c] = get_entity_info(bio_labels, tokens, text=text, spans=spans)
    return task_entities
