import pandas as pd


def load_sentiment_lexicon(path, return_as_dict=True):
    df_lexicon = pd.read_csv(path)
    df_lexicon = (
        df_lexicon[df_lexicon.score.isnull()]
        .drop("score", axis=1)
        .assign(lexicon_label=lambda x: x.sentiment.str.cat(x.lexicon, sep="_"))
    )
    if return_as_dict:
        SENTIMENT_LEXICON = (
            df_lexicon.groupby("word").lexicon_label.apply(list).to_dict()
        )
        return SENTIMENT_LEXICON
    return df_lexicon
