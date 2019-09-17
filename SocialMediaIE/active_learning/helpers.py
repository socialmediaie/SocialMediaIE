import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def get_predictions(df, model, scoring_fn):
    predictions = model.predict_proba(df.text)
    selection_scores = scoring_fn(predictions)
    predictions = pd.DataFrame(predictions, columns=model.classes_, index=df.index)
    predictions["selection_scores"] = selection_scores
    return pd.concat([df, predictions], axis=1)


def classification_metrics(df, model):
    true_labels = df.label
    predicted_labels = model.predict(df.text)
    report = pd.DataFrame(
        classification_report(true_labels, predicted_labels, output_dict=True)
    ).T
    cm = confusion_matrix(true_labels, predicted_labels, labels=model.classes_)
    cm = pd.DataFrame(cm, columns=model.classes_, index=model.classes_)
    return cm, report


def get_joined_metrics(metrics, base_metrics=None):
    all_cms = {i: iter_metrics[0] for i, iter_metrics in enumerate(metrics)}    
    all_reports = {i: iter_metrics[1] for i, iter_metrics in enumerate(metrics)}
    if base_metrics is not None:
        all_cms[-1] = base_metrics[0]
        all_reports[-1] = base_metrics[1]
    df_cm = pd.concat(all_cms, axis=0)
    df_reports = pd.concat(all_reports, axis=0)
    return df_cm, df_reports


def default_selected_indexes(df, prop=0.1, size=None):
    if size is not None:
        seed_size = size
    else:
        seed_size = round(df.shape[0] * prop)
    selected_indexes = np.random.choice(df.index.values, size=seed_size, replace=False)
    return selected_indexes


def plot_metrics(metrics, key="f1-score", base_metric=None):
    df_metrics = pd.concat(
        [
            iter_metrics[1][key]
            .to_frame()
            .T.melt(var_name="metric")
            .assign(iteration=i)
            for i, iter_metrics in enumerate(metrics)
        ],
        axis=0,
    )
    g = sns.FacetGrid(df_metrics, col="metric", sharey=False)
    g.map(plt.plot, "iteration", "value", marker=".")
    if base_metric is not None:
        for ax, c in zip(g.axes.flatten(), g.col_names):
            ax.axhline(y=base_metric[1].loc[c, key], color="k", linestyle="--")
    return df_metrics
