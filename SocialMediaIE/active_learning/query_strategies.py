import numpy as np
from scipy import stats


def entropy_scoring(predictions):
    scores = stats.entropy(predictions.T)
    return scores


def min_margin_scoring(predictions):
    """Difference between the top prediction score and next prediction score
    
    predictions: it is n_labels x n_instances
    
    returns:
        n_instances np.array
    """
    prediction_values = np.sort(predictions.T, axis=0)[-2:, :]
    # Take the negative to allow sorting by descending value
    # Take exp to ensure it is positive for proportional weighing
    margin = np.exp(-(prediction_values[-1] - prediction_values[-2]))
    return margin


def select_top(scores, k, ascending=False):
    selected_indexes = scores.sort_values(ascending=ascending).head(k).index
    return selected_indexes


def select_random(scores, k, weights=None):
    k = np.min([k, scores.shape[0]])
    selected_indexes = scores.sample(n=k, replace=False, weights=weights).index
    return selected_indexes


def select_proportional(scores, k):
    selected_indexes = select_random(scores, k, weights=scores)
    return selected_indexes
