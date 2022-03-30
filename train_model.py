from sentence_transformers import SentenceTransformer, models, datasets
import math
from torch import nn
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
import torch
from torch.utils.data import DataLoader
from sentence_transformers import evaluation
import pandas as pd
from sentence_transformers import models, losses
import numpy as np
import pandas as pd
import logging
import random
import csv
import json
from glob import glob
from sklearn.model_selection import train_test_split
import os
import argparse
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

MODEL_TRANSFORMERS_DIR = {'bertin': 'bertin-project/bertin-roberta-base-spanish'}

def mkmodel(args):
    model_dir = MODEL_TRANSFORMERS_DIR[args.backbone]
    word_embedding_model = models.Transformer(model_dir)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=False,
                                   pooling_mode_cls_token=True,
                                   pooling_mode_max_tokens=False)
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                               out_features=512, #ojimetro
                               activation_function=nn.Tanh())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    model.cuda() 
    return model


def main(args) -> None:
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    model = mkmodel(args)

    train_data = {}
    def add_to_samples(sent1, sent2, label):
        if label in ['contradiction', 'entailment', 'neutral']:
            if sent1 not in train_data:
                train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
            train_data[sent1][label].add(sent2)
        else:
            pass

    # Read the AllNLI.tsv.gz file and create the training dataset
    logging.info("Read ESxNLI train dataset")
    with open('esxnli.tsv') as dataset:
      reader = csv.DictReader(dataset, delimiter='\t', quoting=csv.QUOTE_NONE)
      for row in reader:
        if row['language'] == 'es':
          
          sent1 = row['sentence1'].strip()
          sent2 = row['sentence2'].strip()
        
          add_to_samples(sent1, sent2, row['gold_label'])
          add_to_samples(sent2, sent1, row['gold_label'])  #Also add the opposite

    if not args.only_esxnli:
        logging.info("Read SNLI and MultiNLI train dataset")
        # Add SNLI and MultiNLI
        snli_root = 'datav2'
        snli_files = ['ESsnli_1.0_train.jsonl', 'multinli_1.0_train_translated1.jsonl']

        for fname in snli_files:
            print('Loading: {}'.format(fname))
            with open(os.path.join(snli_root, fname)) as fl:
                for entry in fl:
                    row = json.loads(entry)
                    sent1 = row['sentence1'].strip()
                    sent2 = row['sentence2'].strip()
                    add_to_samples(sent1, sent2, row['gold_label'])
                    add_to_samples(sent2, sent1, row['gold_label'])  #Also add the opposite

    # create dataloader
    train_samples = []
    for sent1, others in train_data.items():
        if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
            train_samples.append(InputExample(texts=[sent1, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]))
            train_samples.append(InputExample(texts=[random.choice(list(others['entailment'])), sent1, random.choice(list(others['contradiction']))]))

    # Special data loader that avoid duplicates within a batch
    train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=64)

    logging.info("Read STS2015 ES testset")
    sts_root = 'sts2015-es'
    sts_files = [('STS.gs.newswire.txt', 'STS.input.newswire.txt'),
                 ('STS.gs.wikipedia.txt', 'STS.input.wikipedia.txt')]

    dev_examples = []
    for gs_fname, sents_fname in sts_files:
        gs = open(os.path.join(sts_root, gs_fname))
        sents = open(os.path.join(sts_root, sents_fname))
        for g, s in zip(gs, sents):
            sent1, sent2 = s.split('\t')
            dp = InputExample(texts=[sent1, sent2], label=float(g)/5)
            dev_examples.append(dp)
        gs.close()
        sents.close()
        
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_examples,
                                                                 batch_size=args.batch_size,
                                                                 name='sts-test')

    # Our training loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * 0.05)
    logging.info("Warmup-steps: {}".format(warmup_steps))

    logging.info("Training")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=args.epochs,
              warmup_steps=warmup_steps,
              evaluator=evaluator,
              output_path=args.output_path,
              show_progress_bar=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='bertin')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--output_path', default='bertin-hugenli')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--only_esxnli', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
