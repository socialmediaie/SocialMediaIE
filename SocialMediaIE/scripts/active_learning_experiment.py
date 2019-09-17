import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from SocialMediaIE.active_learning.helpers import (get_joined_metrics,
                                                   plot_metrics)
from SocialMediaIE.active_learning.query_strategies import (
    entropy_scoring, min_margin_scoring, select_proportional, select_random,
    select_top)
from SocialMediaIE.data.load_lexicon import load_sentiment_lexicon
from SocialMediaIE.data.text_preprocess import (create_lexicon_feature_fn,
                                                preprocess_text)
from SocialMediaIE.models.sklearn_bag_of_words import get_model
from SocialMediaIE.training.active_learning_trainer import \
    ActiveLearningTrainer

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

sns.set_context("talk")
sns.set_style("ticks")
np.random.seed(1337)


def create_trainer(args):
    SENTIMENT_LEXICON = load_sentiment_lexicon(args.lexicon_path)
    model_fn = lambda: get_model(SENTIMENT_LEXICON)
    if args.scoring == "min_margin":
        scoring_fn = min_margin_scoring
    elif args.scoring == "entropy":
        scoring_fn = entropy_scoring
    else:
        raise RuntimeError(f"args.scoring={args.scoring} is invalid")
    if args.selection == "top":
        selection_fn = select_top
    elif args.selection == "random":
        selection_fn = select_random
    elif args.selection == "proportional":
        selection_fn = select_proportional
    else:
        raise RuntimeError(f"args.selection={args.selection} is invalid")
    trainer = ActiveLearningTrainer(
        model_fn, scoring_fn=scoring_fn, selection_fn=selection_fn
    )
    return trainer


def main(args):
    DATA_KEY = args.data_key
    TASK_KEY = args.task_key
    df_train = pd.read_json(
        f"./data/processed/{TASK_KEY}/{DATA_KEY}/train.json",
        orient="records",
        lines=True,
    )
    eval_dfs = {
        k: pd.read_json(
            f"./data/processed/{TASK_KEY}/{DATA_KEY}/{k}.json",
            orient="records",
            lines=True,
        )
        for k in ["dev", "test"]
    }

    trainer = create_trainer(args)
    all_metrics, base_metrics, training_indexes = trainer.train_multiple_rounds(
        df_train,
        eval_dfs=eval_dfs,
        annotations_per_step=args.annotations_per_step,
        max_iterations=args.max_iters,
    )
    output_dir = os.path.join(
        args.output_dir, TASK_KEY, DATA_KEY, f"{args.scoring}_{args.selection}"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    training_indexes.to_csv(os.path.join(output_dir, "training_indexes.csv"))
    for k, metrics in all_metrics.items():
        df_cm, df_reports = get_joined_metrics(metrics, base_metrics=base_metrics.get(k))
        df_cm.to_csv(os.path.join(output_dir, f"{k}_cm.csv"))
        df_reports.to_csv(os.path.join(output_dir, f"{k}_reports.csv"))
        plot_metrics(metrics, base_metric=base_metrics.get(k))
        plt.savefig(os.path.join(output_dir, f"{k}_metrics.pdf"), bbox_inches="tight")


def create_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Active learning experiment")
    parser.add_argument("--data-key", default="SemEval")
    parser.add_argument("--task-key", default="SENTIMENT")
    parser.add_argument(
        "--max-iters", default=100, type=int, help="number of active learning rounds"
    )
    parser.add_argument(
        "--lexicon-path", default="./data/sentiments.csv", help="sentiment lexicon"
    )
    parser.add_argument(
        "--annotations-per-step",
        default=100, type=int,
        help="Number of annotations per step before restarting training.",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/active_learning_models/",
        help="output directory to store model metrics.",
    )
    parser.add_argument(
        "--scoring",
        choices=["entropy", "min_margin"],
        default="entropy",
        help="scoring function",
    )
    parser.add_argument(
        "--selection",
        choices=["random", "top", "proportional"],
        default="top",
        help="scoring function",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print(args)
    main(args)
