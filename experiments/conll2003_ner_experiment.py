from SocialMediaIE.readers import *
from SocialMediaIE.utils import *
from SocialMediaIE.datasets import SentenceDataset
from SocialMediaIE.models.id_cnn_seq import *
from SocialMediaIE.callbacks import EarlyStopping, VisdomMonitor
from SocialMediaIE.callbacks import ModelCheckpointing
from SocialMediaIE.metrics import CONLLEval
import numpy as np
import sys
import logging

def main(args):
    logging.basicConfig(
            stream=sys.stderr,
            level=args.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    logger = logging.getLogger(__name__)

    logger.info(args)
    DATA_FILES=dict(
            TRAIN="/datadrive/Datasets/NLP/CONLL2003/2col/eng.train",
            DEV="/datadrive/Datasets/NLP/CONLL2003/2col/eng.testa",
            TEST="/datadrive/Datasets/NLP/CONLL2003/2col/eng.testb",
            )
    sep=" "
    logger.info(DATA_FILES)
    (
            token_vocab,
            char_vocab,
            label_vocab
            ) = get_vocabs(
                    read_conll_data(DATA_FILES.values(), sep=sep),
                    word_preprocess=twitter_preprocess,
                    char_preprocess=char_preprocess
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

    sentences = get_sentences(
            read_conll_data(DATA_FILES["TRAIN"], sep=sep)
            )
    maxlen = get_maxlen(sentences)
    logger.info("Using maxlen: {}".format(maxlen))
    dataloaders = dict()
    for key, filename in DATA_FILES.items():
        sentences = get_sentences(
                read_conll_data(filename, sep=sep)
                )
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
                char_conv_kernel=3,
                dropout=0.5,
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
                decoder_layers=3,
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--min-delta", type=float, default=1e-3, help="minimum delta for early stopping")
    parser.add_argument("--patience", type=int, default=10, help="patience for early stopping")

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
    args = parser.parse_args()
    main(args)
