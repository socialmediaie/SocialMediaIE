import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from SocialMediaIE.data.load_lexicon import load_sentiment_lexicon
from SocialMediaIE.data.text_preprocess import (create_joined_feature_fn,
                                                create_lexicon_feature_fn,
                                                preprocess_text)
from SocialMediaIE.data.tokenization import tokenize


def get_model(lexicon=None):
    lexicon_feature = None
    if lexicon is not None:
        lexicon_feature = create_lexicon_feature_fn(lexicon)
    tokenizer = create_joined_feature_fn(preprocess_text, lexicon_feature)
    steps = [
        ("TfidfVectorizer", TfidfVectorizer(tokenizer=tokenizer)),
        (
            "model",
            LogisticRegressionCV(solver="lbfgs", multi_class="multinomial", n_jobs=-1),
        ),
    ]
    model = Pipeline(steps)
    return model
