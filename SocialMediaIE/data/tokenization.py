import re
import pandas as pd

# -*- coding: utf-8 -*-

# Taken from https://github.com/bfelbo/DeepMoji/blob/master/deepmoji/tokenizer.py
"""
Splits up a Unicode string into a list of tokens.
Recognises:
- Abbreviations
- URLs
- Emails
- #hashtags
- @mentions
- emojis
- emoticons (limited support)

Multiple consecutive symbols are also treated as a single token.
"""

# Basic patterns.
RE_NUM = r"[0-9]+"
RE_WORD = r"[a-zA-Z]+"
RE_WHITESPACE = r"\s+"
RE_ANY = r"."

# Combined words such as 'red-haired' or 'CUSTOM_TOKEN'
RE_COMB = r"[a-zA-Z]+[-_][a-zA-Z]+"

# English-specific patterns
RE_CONTRACTIONS = RE_WORD + r"\'" + RE_WORD

TITLES = [r"Mr\.", r"Ms\.", r"Mrs\.", r"Dr\.", r"Prof\."]
# Ensure case insensitivity
RE_TITLES = r"|".join([r"(?i)" + t for t in TITLES])

# Symbols have to be created as separate patterns in order to match consecutive
# identical symbols.
SYMBOLS = r"()<!?.,/\'\"-_=\\§|´ˇ°[]<>{}~$^&*;:%+\xa3€`"
RE_SYMBOL = r"|".join([re.escape(s) + r"+" for s in SYMBOLS])

# Hash symbols and at symbols have to be defined separately in order to not
# clash with hashtags and mentions if there are multiple - i.e.
# ##hello -> ['#', '#hello'] instead of ['##', 'hello']
SPECIAL_SYMBOLS = r"|#+(?=#[a-zA-Z0-9_]+)|@+(?=@[a-zA-Z0-9_]+)|#+|@+"
RE_SYMBOL += SPECIAL_SYMBOLS

RE_ABBREVIATIONS = r"\b(?<!\.)(?:[A-Za-z]\.){2,}"

# Twitter-specific patterns
RE_HASHTAG = r"#[a-zA-Z0-9_]+"
RE_MENTION = r"@[a-zA-Z0-9_]+"

RE_URL = r"(?:https?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
RE_EMAIL = r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"

# Emoticons and emojis
RE_HEART = r"(?:<+/?3+)+"
EMOTICONS_START = [r">:", r":", r"=", r";"]
EMOTICONS_MID = [r"-", r",", r"^", "'", '"']
EMOTICONS_END = [
    r"D",
    r"d",
    r"p",
    r"P",
    r"v",
    r")",
    r"o",
    r"O",
    r"(",
    r"3",
    r"/",
    r"|",
    "\\",
]
EMOTICONS_EXTRA = [r"-_-", r"x_x", r"^_^", r"o.o", r"o_o", r"(:", r"):", r");", r"(;"]

RE_EMOTICON = r"|".join([re.escape(s) for s in EMOTICONS_EXTRA])
for s in EMOTICONS_START:
    for m in EMOTICONS_MID:
        for e in EMOTICONS_END:
            RE_EMOTICON += "|{0}{1}?{2}+".format(
                re.escape(s), re.escape(m), re.escape(e)
            )

# requires ucs4 in python2.7 or python3+
# RE_EMOJI = r"""[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]"""
# safe for all python
RE_EMOJI = r"""\ud83c[\udf00-\udfff]|\ud83d[\udc00-\ude4f\ude80-\udeff]|[\u2600-\u26FF\u2700-\u27BF]"""

# List of matched token patterns, ordered from most specific to least specific.
TOKENS = [
    RE_URL,
    RE_EMAIL,
    # RE_COMB,
    RE_HASHTAG,
    RE_MENTION,
    RE_HEART,
    RE_EMOTICON,
    RE_CONTRACTIONS,
    RE_TITLES,
    RE_ABBREVIATIONS,
    RE_NUM,
    RE_WORD,
    RE_SYMBOL,
    RE_EMOJI,
    RE_ANY,
]

# List of ignored token patterns
IGNORED = [RE_WHITESPACE]

# Final pattern
RE_PATTERN = re.compile(
    r"(?P<space>" + r"|".join(IGNORED) + r")|(?P<token>" + r"|".join(TOKENS) + r")",
    re.UNICODE | re.MULTILINE,
)

RE_URL = re.compile(RE_URL, re.UNICODE)
RE_EMOJI = re.compile(RE_EMOJI, re.UNICODE)
RE_EMOTICON = re.compile(RE_EMOTICON, re.UNICODE)
RE_SYMBOL = re.compile(RE_SYMBOL, re.UNICODE)
RE_HASHTAG = re.compile(RE_HASHTAG, re.UNICODE)
RE_MENTION = re.compile(RE_MENTION, re.UNICODE)


def tokenize(text):
    """Splits given input string into a list of tokens.

    # Arguments:
        text: Input string to be tokenized.

    # Returns:
        List of strings (tokens).
    """
    result = RE_PATTERN.findall(text)

    # Remove empty strings
    result = [t[1] for t in result if t[1].strip()]
    return result


def get_match_iter(text):
    return (match for match in RE_PATTERN.finditer(text))


def get_match_object(match):
    text_type, text_value = [
        (k, v) for k, v in match.groupdict().items() if v is not None
    ][0]
    obj = {
        "type": text_type,
        "value": text_value,
        "span": match.span(text_type),
        "is_hashtag": RE_HASHTAG.match(text_value) is not None,
        "is_mention": RE_MENTION.match(text_value) is not None,
        "is_url": RE_URL.match(text_value) is not None,
        "is_emoji": RE_EMOJI.match(text_value) is not None,
        "is_emoticon": RE_EMOTICON.match(text_value) is not None,
        "is_symbol": len(text_value) == 1 and RE_SYMBOL.match(text_value) is not None,
    }
    return obj


def get_token_objects(text):
    objects = [get_match_object(match) for match in get_match_iter(text)]
    num_tokens = len(objects)
    cleaned_objects = []
    for i, obj in enumerate(objects):
        obj["no_space"] = True
        obj["token_idx"] = i
        if obj["type"] == "space":
            continue
        if i < num_tokens-1 and objects[i+1]["type"] == "space":
            obj["no_space"] = False
        cleaned_objects.append(obj)
    keys = cleaned_objects[0].keys()
    final_sequences = {}
    for k in keys:
        final_sequences[k] = [obj[k] for obj in cleaned_objects]
    return final_sequences

COLUMN_ORDER = [
    "value",
    "token_idx",
    "type",
    "span",
    "is_hashtag",
    "is_mention",
    "is_url",
    "is_emoji",
    "is_emoticon",
    "is_symbol",
    "no_space"
]


def tokens_as_df(text):
    token_objects = get_token_objects(text)
    df = pd.DataFrame(token_objects)[COLUMN_ORDER]
    return df
