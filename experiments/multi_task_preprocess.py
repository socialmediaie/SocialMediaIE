
# coding: utf-8

# In[1]:

from glob import glob
from collections import Counter
import pandas as pd

from IPython.display import display


# In[2]:

NER_FILES={
    "Finin": {
        "train": "/datadrive/Datasets/lowlands-data/LREC2014/twitter_ner/data/finin.train.tsv",
        "test": "/datadrive/Datasets/lowlands-data/LREC2014/twitter_ner/data/finin.test.tsv.utf8",
    },
    "Hege": {
        "test": "/datadrive/Datasets/lowlands-data/LREC2014/twitter_ner/data/hege.test.tsv",
    },
    "Ritter": {
        "test": "/datadrive/Datasets/lowlands-data/LREC2014/twitter_ner/data/ritter.test.tsv",
    },
    "WNUT_2016": {
        "train": "/datadrive/Codes/multi-task-nlp-keras/data/WNUT_NER/train.tsv",
        "test": "/datadrive/Codes/multi-task-nlp-keras/data/WNUT_NER/test.tsv",
        "dev": "/datadrive/Codes/multi-task-nlp-keras/data/WNUT_NER/dev.tsv",
    },
    "WNUT_2017": {
        "train": "/datadrive/Codes/multi-task-nlp-keras/data/WNUT_2017/wnut17train.conll",
        "dev": "/datadrive/Codes/multi-task-nlp-keras/data/WNUT_2017/emerging.dev.conll",
        "test": "/datadrive/Codes/multi-task-nlp-keras/data/WNUT_2017/emerging.test.annotated",
    },
    "MSM_2013": {
        "train": "/datadrive/Datasets/Twitter/MSM2013/data/msm2013-ce_challenge_gs/TweetsTrainingSetCH.tsv.conll",
        "test": "/datadrive/Datasets/Twitter/MSM2013/data/msm2013-ce_challenge_gs/goldStandard.tsv.conll",
    }
}


POS_FILES={
    "Owoputi_2013": {
        "train": "/datadrive/Datasets/Twitter/TweeboParser/ark-tweet-nlp-0.3.2/data/twpos-data-v0.3/oct27.splits/oct27.train",
        "traindev": "/datadrive/Datasets/Twitter/TweeboParser/ark-tweet-nlp-0.3.2/data/twpos-data-v0.3/oct27.splits/oct27.traindev",
        "dev": "/datadrive/Datasets/Twitter/TweeboParser/ark-tweet-nlp-0.3.2/data/twpos-data-v0.3/oct27.splits/oct27.dev",
        "test": "/datadrive/Datasets/Twitter/TweeboParser/ark-tweet-nlp-0.3.2/data/twpos-data-v0.3/oct27.splits/oct27.test",
        "daily547": "/datadrive/Datasets/Twitter/TweeboParser/ark-tweet-nlp-0.3.2/data/twpos-data-v0.3/daily547.conll"
    },
    "LexNorm_Li_2015": {
        "dev": "/datadrive/Datasets/Twitter/wnut-2017-pos-norm/data/test_L.gold"        
    },
    ## Next 3 use Universal POS mappings: 
    "Foster_2011": {
        "test": "/datadrive/Datasets/lowlands-data/ACL2014/crowdsourced_POS/data/foster-twitter.test"
    },
    "Ritter_2011": {
        "test": "/datadrive/Datasets/lowlands-data/ACL2014/crowdsourced_POS/data/ritter.test"
    },
    "lowlands": {
        "test": "/datadrive/Datasets/lowlands-data/ACL2014/crowdsourced_POS/data/lowlands.test"
    },
    "Gimple_2012": {
        "test": "/datadrive/Datasets/lowlands-data/ACL2014/crowdsourced_POS/data/gimpel.GOLD"
    },
    "Bootstrap_2013": {
        # Full PTB tagset, plus four custom tags (USR, HT, RT, URL)
        "train": "/datadrive/Datasets/Twitter/twitter-pos-bootstrap/data/bootstrap.conll"
    }
}

SENTIMENT_FILES={
    "SMILE": {
        "train": "/datadrive/Datasets/Twitter/SMILE/smile-annotations-final.csv",
    },
}

SUPERSENSE_TAGGING_FILES={
    "Ritter": {
        "train": "/datadrive/Datasets/Twitter/supersense-data-twitter/ritter-train.tsv",
        "dev": "/datadrive/Datasets/Twitter/supersense-data-twitter/ritter-dev.tsv",
        "test": "/datadrive/Datasets/Twitter/supersense-data-twitter/ritter-eval.tsv"
    },
    "Johannsen_2014": {
        "test": "/datadrive/Datasets/Twitter/supersense-data-twitter/in-house-eval.tsv"
    }
}

FRAME_SEMANTICS_FILE={
    "Sogaard_2015": {
        "gavin": "/datadrive/Datasets/lowlands-data/AAAI15/conll/all.gavin",
        "maria": "/datadrive/Datasets/lowlands-data/AAAI15/conll/all.maria",
        "sara": "/datadrive/Datasets/lowlands-data/AAAI15/conll/all.sara"
    }
}

DIMSUM_FILES = {
    # Following data is already part of dimsum
    #"Lowlands": {
    #    "test": "/datadrive/Datasets/Twitter/dimsum-data/conversion/original/lowlands.UPOS2.tsv"
    #},
    #"Ritter": {
    #    "test": "/datadrive/Datasets/Twitter/dimsum-data/conversion/original/ritter.UPOS2.tsv"
    #},
    #"Streusle": {
    #    "test": "/datadrive/Datasets/Twitter/dimsum-data/conversion/original/streusle.upos.tags"
    #}, 
    "DiMSUM_2016": {
        # Made in combination with ritter, streusle, lowlands
        # 55579 ewtb
        # 3062 lowlands
        # 15185 ritter
        "train": "/datadrive/Datasets/Twitter/dimsum-data/conll/dimsum16.train",
        # 3516 ted
        # 6357 trustpilot
        # 6627 tweebank
        "test": "/datadrive/Datasets/Twitter/dimsum-data/conll/dimsum16.test"
    }
}

PARSING_FILES={
    "Kong_2014": {
        "train": "/datadrive/Datasets/Twitter/TweeboParser/Tweebank/Train_Test_Splited/train",
        "test": "/datadrive/Datasets/Twitter/TweeboParser/Tweebank/Train_Test_Splited/test",
    }
}

WEB_TREEBANK={
    "DenoisedWebTreebank": {
        "dev": "/datadrive/Datasets/Twitter/DenoisedWebTreebank/data/DenoisedWebTreebank/dev.conll",
        "test": "/datadrive/Datasets/Twitter/DenoisedWebTreebank/data/DenoisedWebTreebank/test.conll"
    }
}

NORMALIZED={
    "DenoisedWebTreebank": {
        "dev": "/datadrive/Datasets/Twitter/DenoisedWebTreebank/data/DenoisedWebTreebank/dev.normalized",
        "test": "/datadrive/Datasets/Twitter/DenoisedWebTreebank/data/DenoisedWebTreebank/test.normalized"
    }
}

PARAPHRASE_SEMANTIC_FILES={
    "SemEval-2015 Task 1": {
        #  Topic_Id | Topic_Name | Sent_1 | Sent_2 | Label | Sent_1_tag | Sent_2_tag |
        # Map labels as follows
        # paraphrases: (3, 2) (4, 1) (5, 0)
        # non-paraphrases: (1, 4) (0, 5)
        # debatable: (2, 3)  which you may discard if training binary classifier
        
        "train": "/datadrive/Datasets/Twitter/SemEval-PIT2015-github/data/train.data",
        "dev": "/datadrive/Datasets/Twitter/SemEval-PIT2015-github/data/dev.data",
        "test": "/datadrive/Datasets/Twitter/SemEval-PIT2015-github/data/test.data"
    }
}


# In[3]:

def read_conll_data(filename, ncols=2):
    with open(filename, encoding='utf-8') as fp:
        for seq in fp.read().split("\n\n"):
            seq_ = []
            for line in seq.splitlines():
                line = line.rstrip()
                if not line:
                    continue
                values = line.split("\t")
                if len(values) < ncols:
                    # Skip invalid lines
                    continue
                seq_.append(values)
            if not seq_:
                seq_ = []
                continue
            yield seq_
                


# In[4]:

def get_ner_label(label, idx=1):
    if label.upper() == "O":
        return label
    if idx is None:
        return label
    return label.split('-', 1)[idx]

def get_simple_label(label):
    if label:
        return label
    return "O"

def get_file_stats(
    filename,
    label_processor=None,
    label_col_id=-1,
    skip_other=True,
    ncols=2
):
    if label_processor is None:
        label_processor = lambda x: x
    total_seq = 0
    total_tokens = 0
    token_types = Counter()
    for i, seq in enumerate(read_conll_data(filename, ncols=ncols)):
        total_seq += 1
        total_tokens += len(seq)
        try:
            for item in seq:
                label = label_processor(item[label_col_id])
                if skip_other and label == "O":
                    continue
                token_types.update([
                    label
                ])
        except IndexError:
            print(i, seq)
            raise
    return total_seq, total_tokens, token_types
        


# In[5]:

def make_conll_dataset_tables(files, **kwargs):
    all_stats = []
    for datakey in files:
        for datatype, filepath in files[datakey].items():
            print("{}-{}: {}".format(datakey, datatype, filepath))
            total_seq, total_tokens, token_types = get_file_stats(filepath, **kwargs)
            print(total_seq, total_tokens, token_types)
            all_stats.append((datakey, datatype, total_seq, total_tokens, token_types))
    return all_stats


def generate_tables(files, display_df=False, show_labels=True, **kwargs):
    all_stats = make_conll_dataset_tables(files, **kwargs)
    df = pd.DataFrame(all_stats, columns=[
        "datakey", "datatype", "total_seq", "total_tokens", "labels"])
    if show_labels:
        df = df.assign(
            all_labels=df["labels"].apply(lambda x: (", ".join(sorted(x.keys()))).upper())
        )
    df = df.assign(
        num_labels=df["labels"].apply(len),
    ).sort_values(["datakey", "datatype"])
    if display_df:
        display(df)
    with pd.option_context("display.max_colwidth", -1):
        print(df.drop("labels", 1).set_index(["datakey", "datatype"]).to_latex())
        display(df.drop("labels", 1).set_index(["datakey", "datatype"]))
    


# In[6]:

generate_tables(NER_FILES, display_df=True, label_processor=lambda x: get_ner_label(x, idx=1))


# ## POS datasets

# In[7]:

generate_tables(POS_FILES, display_df=False)


# ## Supersense tagging

# In[8]:

generate_tables(SUPERSENSE_TAGGING_FILES, display_df=False)


# In[9]:

generate_tables(SUPERSENSE_TAGGING_FILES, label_processor=lambda x: get_ner_label(x, idx=1))


# ## DimSUM 
# 
# https://dimsum16.github.io/

# In[10]:

generate_tables(DIMSUM_FILES, label_col_id=7, label_processor=get_simple_label, skip_other=True)


# ## Frame Semantics
# 
# 
# 
# ```
# @paper{AAAI159349,
# 	author = {Anders Søgaard and Barbara Plank and Hector Alonso},
# 	title = {Using Frame Semantics for Knowledge Extraction from Twitter},
# 	conference = {AAAI Conference on Artificial Intelligence},
# 	year = {2015},
# 	keywords = {frame semantics; knowledge bases; twitter},
# 	abstract = {Knowledge bases have the potential to advance artificial intelligence, but often suffer from recall problems, i.e., lack of knowledge of new entities and relations. On the contrary, social media such as Twitter provide abundance of data, in a timely manner: information spreads at an incredible pace and is posted long before it makes it into more commonly used resources for knowledge extraction. In this paper we address the question whether we can exploit social media to extract new facts, which may at first seem like finding needles in haystacks. We collect tweets about 60 entities in Freebase and compare four methods to extract binary relation candidates, based on syntactic and semantic parsing and simple mechanism for factuality scoring. The extracted facts are manually evaluated in terms of their correctness and relevance for search. We show that moving from bottom-up syntactic or semantic dependency parsing formalisms to top-down frame-semantic processing improves the robustness of knowledge extraction, producing more intelligible fact candidates of better quality. In order to evaluate the quality of frame semantic parsing on Twitter intrinsically, we make a multiply frame-annotated dataset of tweets publicly available.},
# 
# 	url = {https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9349}
# }
# 
# ```

# In[11]:

generate_tables(FRAME_SEMANTICS_FILE, show_labels=False, label_col_id=3, label_processor=get_simple_label, skip_other=True)


# In[ ]:



