from SocialMediaIE.data.tokenization import tokenize


def process_token(token):
    token = token.lower()
    if token.startswith("@"):
        return "@USER"
    if token.startswith("#"):
        return "#HASHTAG"
    if token.startswith(("http://", "https://", "www.")):
        return "http://URL.COM"
    return token


def preprocess_text(text):
    tokens = [process_token(token) for token in tokenize(text)]
    return tokens


def create_lexicon_feature_fn(lexicon):
    _, default_value = next((item for item in lexicon.items()))

    def get_lexicon_features(tokens):
        if isinstance(default_value, str):
            lexicon_items = filter(lambda x: x, [lexicon.get(k) for k in tokens])
        else:
            lexicon_items = sum([lexicon.get(k, []) for k in tokens], [])
        return lexicon_items

    return get_lexicon_features


def create_joined_feature_fn(token_fn, *other_fns):
    def get_features(text):
        tokens = token_fn(text)
        output = tokens
        for other_fn in other_fns:
            output += other_fn(tokens)
        return output

    return get_features
