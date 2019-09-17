from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from SocialMediaIE.active_learning.helpers import (classification_metrics,
                                                   default_selected_indexes,
                                                   get_predictions)
from SocialMediaIE.active_learning.query_strategies import (entropy_scoring,
                                                            select_top)


class ActiveLearningTrainer(object):
    def __init__(self, model_fn, scoring_fn=entropy_scoring, selection_fn=select_top):
        self.model_fn = model_fn
        self.scoring_fn = scoring_fn
        self.selection_fn = selection_fn

    def get_base_metrics(self, df_train, eval_dfs=None):
        if eval_dfs is None:
            eval_dfs = {}
        eval_dfs["train"] = df_train
        eval_dfs = {k:v for k,v in eval_dfs.items()}
        model = self.model_fn()
        model.fit(df_train.text, df_train.label)
        metrics = self.get_metrics(model, eval_dfs)
        return metrics

    def get_metrics(self, model, eval_dfs):
        metrics = {}
        for key, df_eval in eval_dfs.items():
            metrics[key] = classification_metrics(df_eval, model)
        return metrics

    def train_single_round(self, df_train, eval_dfs=None):
        df_selected = df_train[df_train.selected]
        df_unselected = df_train[~df_train.selected]
        # Fit model
        model = self.model_fn()
        # TODO: Make the model fit inputs more modular
        model.fit(df_selected.text, df_selected.label)
        if eval_dfs is None:
            eval_dfs = {}
        eval_dfs = {k:v for k,v in eval_dfs.items()}
        eval_dfs["train"] = df_train
        if df_unselected.shape[0] > 0:
            eval_dfs["unselected"] = df_unselected
        metrics = self.get_metrics(model, eval_dfs)
        return model, metrics, df_unselected

    def train_multiple_rounds(
        self,
        df_train,
        eval_dfs=None,
        selected_indexes=None,
        default_selected_prop=0.1,
        annotations_per_step=100,
        max_iterations=10,
    ):
        df_train["selected"] = False
        if selected_indexes is None:
            selected_indexes = default_selected_indexes(
                df_train, prop=default_selected_prop, size=annotations_per_step
            )
        df_train.loc[selected_indexes, "selected"] = True
        training_indexes = pd.DataFrame({"selected_indexes": selected_indexes}).assign(
            epoch=-1
        )
        all_metrics = defaultdict(list)
        # Train a round
        with tqdm(
            desc=f"Num train={training_indexes.shape[0]}",
            total=max_iterations + 1,
            unit="epoch",
        ) as pbar:
            for i in range(max_iterations):
                model, round_metrics, df_unselected = self.train_single_round(df_train, eval_dfs=eval_dfs)
                if df_unselected.shape[0] == 0:
                    pbar.set_postfix({"Breaking round": i})
                    break
                for k, v in round_metrics.items():
                    all_metrics[k].append(v)
                # Select new data
                df_pred = get_predictions(df_unselected, model, self.scoring_fn)
                # Identify top instances to label
                selected_indexes = self.selection_fn(df_pred["selection_scores"], annotations_per_step)
                # selected_indexes = df_pred.sort_values("selection_scores", ascending=False).head(annotations_per_step).index
                # Add these instances to the training data
                training_indexes = pd.concat(
                    [
                        training_indexes,
                        pd.DataFrame({"selected_indexes": selected_indexes}).assign(
                            epoch=i
                        ),
                    ],
                    axis=0,
                )
                df_train.loc[selected_indexes, "selected"] = True
                # print(f"Selected data distribution: {df_pred.loc[new_indexes].label.value_counts().to_dict()}")
                # print(f"Total training size: {df_train[df_train.selected].shape[0]}")
                pbar.set_description(f"Num train={training_indexes.shape[0]}")
                selected_label_distribution = df_pred.loc[selected_indexes].label.value_counts().to_dict()
                pbar.set_postfix(selected_label_distribution)
                pbar.update(1)
            # Fit model on final selected data and get metrics
            model, round_metrics, df_unselected = self.train_single_round(df_train, eval_dfs=eval_dfs)
            for k, v in round_metrics.items():
                all_metrics[k].append(v)
            pbar.update(1)
            pbar.set_postfix({
                "unselected": df_unselected.shape[0],
                "selected": df_train[df_train.selected].shape[0],
                "total": df_train.shape[0],
            })
        # Get base metrics
        base_metrics = self.get_base_metrics(df_train, eval_dfs)
        return all_metrics, base_metrics, training_indexes
