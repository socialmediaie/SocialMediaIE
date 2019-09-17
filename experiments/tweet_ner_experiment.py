from SocialMediaIE.readers import *
from SocialMediaIE.utils import *
from SocialMediaIE.datasets import SentenceDataset
from SocialMediaIE.models.id_cnn_seq import *
from SocialMediaIE.callbacks import EarlyStopping, VisdomMonitor
from SocialMediaIE.callbacks import ModelCheckpointing
from SocialMediaIE.metrics import CONLLEval
from torch.utils.data import DataLoader
import numpy as np
import sys
import logging

ALL_DATA_FILES=dict(
        TRAIN=[
            "/datadrive/Datasets/lowlands-data/LREC2014/twitter_ner/data/finin.train.tsv",
            "/datadrive/Datasets/Twitter/MSM2013/data/msm2013-ce_challenge_gs/TweetsTrainingSetCH.tsv.conll",
            "/datadrive/Codes/multi-task-nlp-keras/data/WNUT_NER/train.tsv",
            "/datadrive/Codes/multi-task-nlp-keras/data/WNUT_NER/dev.tsv",
            "/datadrive/Codes/multi-task-nlp-keras/data/WNUT_2017/wnut17train.conll.tsv",
            "/datadrive/Codes/multi-task-nlp-keras/data/WNUT_2017/emerging.dev.conll.tsv"
            ],
        DEV="SplitTrain20",
        FININ="/datadrive/Datasets/lowlands-data/LREC2014/twitter_ner/data/finin.test.tsv.utf8",
        MSM2013="/datadrive/Datasets/Twitter/MSM2013/data/msm2013-ce_challenge_gs/goldStandard.tsv.conll",
        WNUT2016="/datadrive/Codes/multi-task-nlp-keras/data/WNUT_NER/test.tsv",
        WNUT2017="/datadrive/Codes/multi-task-nlp-keras/data/WNUT_2017/emerging.test.annotated.tsv",
        RITTER="/datadrive/Datasets/lowlands-data/LREC2014/twitter_ner/data/ritter.test.tsv",
        HEGE="/datadrive/Datasets/lowlands-data/LREC2014/twitter_ner/data/hege.test.tsv",
        )

## FININ is also called UMBC dataset
FININ_DATA_FILES=dict(
        TRAIN="/datadrive/Datasets/lowlands-data/LREC2014/twitter_ner/data/finin.train.tsv",
        DEV="SplitTrain20",
        TEST="/datadrive/Datasets/lowlands-data/LREC2014/twitter_ner/data/finin.test.tsv.utf8",
        )

MSM2013_DATA_FILES=dict(
        TRAIN="/datadrive/Datasets/Twitter/MSM2013/data/msm2013-ce_challenge_gs/TweetsTrainingSetCH.tsv.conll",
        DEV="SplitTrain20",
        TEST="/datadrive/Datasets/Twitter/MSM2013/data/msm2013-ce_challenge_gs/goldStandard.tsv.conll",
        )

WNUT2016_DATA_FILES=dict(
        TRAIN="/datadrive/Codes/multi-task-nlp-keras/data/WNUT_NER/train.tsv",
        DEV="/datadrive/Codes/multi-task-nlp-keras/data/WNUT_NER/dev.tsv",
        TEST="/datadrive/Codes/multi-task-nlp-keras/data/WNUT_NER/test.tsv",
        )
WNUT2017_DATA_FILES=dict(
        TRAIN="/datadrive/Codes/multi-task-nlp-keras/data/WNUT_2017/wnut17train.conll.tsv",
        DEV="/datadrive/Codes/multi-task-nlp-keras/data/WNUT_2017/emerging.dev.conll.tsv",
        TEST="/datadrive/Codes/multi-task-nlp-keras/data/WNUT_2017/emerging.test.annotated.tsv",
        )

def main(args):
    logging.basicConfig(
            stream=sys.stderr,
            level=args.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    logger = logging.getLogger(__name__)

    logger.info(args)
    DATA_FILES = WNUT2016_DATA_FILES
    if args.data == "wnut17":
        DATA_FILES = WNUT2017_DATA_FILES
    if args.data == "finin":
        DATA_FILES = FININ_DATA_FILES
    if args.data == "msm":
        DATA_FILES = MSM2013_DATA_FILES
    if args.data == "all":
        DATA_FILES = ALL_DATA_FILES
    logger.info(DATA_FILES)
    create_dev = False
    if "SplitTrain20" in list(DATA_FILES.values()):
        create_dev = True
        DATA_FILES["DEV"] = DATA_FILES["TRAIN"]
    (
            token_vocab,
            char_vocab,
            label_vocab
            ) = get_vocabs(
                    read_conll_data(DATA_FILES["TRAIN"]),
                    word_preprocess=twitter_preprocess,
                    #char_preprocess=char_preprocess,
                    label_preprocess=ner_label_normalized,
                    )
    if args.emb_file:
        token_vocab.read_embedding_file(
                args.emb_file,
                args.word_emb_size,
                cache_file=args.cache_file
                )
    if args.char_emb_file:
        char_vocab.read_embedding_file(
                args.char_emb_file,
                args.char_emb_size,
                cache_file=args.char_cache_file
                )
    logger.info(token_vocab)
    logger.info(char_vocab)
    logger.info(label_vocab)
    logger.info(label_vocab.idx2token)

    sentences = get_sentences(
            read_conll_data(DATA_FILES["TRAIN"])
            )
    maxlen = get_maxlen(sentences)
    if create_dev:
        shuffled_idx = np.arange(len(sentences))
        np.random.shuffle(shuffled_idx)
        train_idx = shuffled_idx[:int(shuffled_idx.shape[0]*0.8)]
        dev_idx = shuffled_idx[int(shuffled_idx.shape[0]*0.8):]
        logger.info("Creating dev data")
    logger.info("Using maxlen: {}".format(maxlen))
    dataloaders = dict()
    print(DATA_FILES)
    for key, filename in DATA_FILES.items():
        sentences = get_sentences(
                read_conll_data(filename)
                )
        if create_dev:
            if key == "TRAIN":
                sentences = [sentences[idx] for idx in train_idx]
            if key == "DEV":
                sentences = [sentences[idx] for idx in dev_idx]
        sent_dataset = SentenceDataset(
                sentences,
                transform,
                token_vocab,
                char_vocab,
                label_vocab,
                maxlen=maxlen
                )
        dataloader = DataLoader(
                sent_dataset,
                batch_size=32,
                shuffle=True,
                num_workers=3,
                pin_memory=True
                )
        dataloaders[key] = dataloader
    logger.info("Created data loaders for [{}]".format(
        list(dataloaders.keys())
    ))
    word_emb_size = token_vocab.ndims or 128
    logger.debug("Word embedding size: {}".format(word_emb_size))
    model = IDCNNModel(
            char_model_kwargs=dict(
                char_vocab_size=char_vocab.size,
                char_emb_size=args.char_emb_size,
                char_conv_features=64,
                char_conv_kernel=5,
                dropout=0.3,
                pretrained_embeddings=char_vocab.embeddings,
                fix_weights=args.fix_char_emb
                ),
            word_model_kwargs=dict(
                vocab_size=token_vocab.size,
                word_emb_size=word_emb_size,
                pretrained_embeddings=token_vocab.embeddings,
                fix_weights=args.fix_word_emb
                ),
            encoder_kwargs=dict(
                dropout=0.5,
                ),
            decoder_kwargs=dict(
                num_classes=label_vocab.size,
                decoder_layers=1,
                decoder_dropout=0.1
                ),
            min_loss=args.min_loss
            )

    logger.debug("Created model:\n{}".format(model))
    metrics=[
            CONLLEval(
                label_vocab,
                maxlen=maxlen,
                write_to_files=False
                )
            ]
    logger.debug("Created metrics:\n{}".format(metrics))
    callbacks = [
            EarlyStopping(
                monitor="conlleval_f1_DEV",
                min_delta=args.min_delta,
                patience=args.patience,
                verbose=1,
                mode="max"
                ),
            VisdomMonitor(
                logging_dict={
                    "conlleval_f1": [
                        ("conlleval_f1_TRAIN", "Train"),
                        ("conlleval_f1_DEV", "Dev"),
                        ],
                    "loss": [
                        ("mean_loss_TRAIN","Train"),
                        ("mean_loss_DEV", "Dev"),
                        ]
                    },
                port=9999
                )
            ]
    if args.checkpoint_file is not None:
        callbacks.append(
                ModelCheckpointing(
                   args.checkpoint_file,
                   monitor="conlleval_f1_DEV",
                   mode="max"
                    )
                )
    logger.debug("Created callbacks:\n{}".format(callbacks))
    logger.debug("Starting training:")
    model.train(
            dataloaders["TRAIN"],
            args.num_epochs,
            optimizer_class=torch.optim.Adam,
            cuda=True,
            metrics=metrics,
            eval_dataloaders=dict(
                DEV=dataloaders["DEV"]
                ),
            callbacks=callbacks,
            optimizer_kwargs={
                "weight_decay": args.weight_decay,
                }
            )

    print("Final evaluations: ")
    if args.checkpoint_file is not None:
        logger.info("Loading best model from checkpoint file: {}".format(
            args.checkpoint_file
            ))
        checkpoint = torch.load(args.checkpoint_file)
        model.load_state_dict(checkpoint["state_dict"])
    for key, dataloader in dataloaders.items():
        eval_history = model.eval(
                dataloader,
                cuda=True,
                metrics=metrics,
                title=key
                )
        print('[{:>10s}]\t{}'.format(
            key,
            ", ".join([
                "{:>20s}: {:.4f}".format(k,v)
                for k,v in sorted(
                    eval_history.items(),
                    key=lambda x: x[0].rsplit("_")[0]
                    )
                ])
            )
            )


def create_output_dir(args):
    import os
    output_dir = "outputs/{}".format(args.data)
    try:
        os.mkdir(output_dir)
        logger.info("Created directory {}.".format(output_dir))
    except FileExistsError:
        logger.info("{} directory exists.".format(output_dir))
    for key in ["cache_file", "char_cache_file", "checkpoint_file"]:
        value = getattr(args, key)
        if value is not None:
            value = "{}/{}".format(output_dir, value)
            setattr(args, key, value)
    return args

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--min-delta", type=float, default=1e-3, help="minimum delta for early stopping")
    parser.add_argument("--patience", type=int, default=10, help="patience for early stopping")
    parser.add_argument("--decoder-type", type=str, default="id_cnn", help="decoder type ['id_cnn', 'attention']")

    parser.add_argument("--word-emb-size", type=int, default=128, help="size of word embeddings")
    parser.add_argument("--emb-file", type=str, default=None, help="file for pre-trained embeddings")
    parser.add_argument("--cache-file", type=str, default=None, help="file for caching pre-trained embeddings")
    parser.add_argument("--fix-word-emb", type=bool, default=False, help="fix word embeddings")

    parser.add_argument("--char-emb-size", type=int, default=32, help="size of char embeddings")
    parser.add_argument("--char-emb-file", type=str, default=None, help="file for pre-trained char embeddings")
    parser.add_argument("--char-cache-file", type=str, default=None, help="file for caching pre-trained char embeddings")
    parser.add_argument("--fix-char-emb", type=bool, default=False, help="fix char embeddings")

    parser.add_argument("--min-loss", type=bool, default=False, help="Min loss per token across decoded layers")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--log-level", type=int, default=logging.INFO, help="logging level")
    parser.add_argument("--checkpoint-file", type=str, default=None, help="file to save model checkpoints")
    parser.add_argument("--data", type=str, default="wnut16", help="wnut16 or wnut17 or finin or msm or all")
    args = parser.parse_args()
    print(args)
    args = create_output_dir(args)
    main(args)
