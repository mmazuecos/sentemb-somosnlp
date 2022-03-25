# Spanish Sentence Embedding - SomosNLP

Repository for Spanish Sentence Embedding group of SomosNLP hackathon.

## Datasets:
```
# get ESXNLI
wget https://raw.githubusercontent.com/artetxem/esxnli/master/esxnli.tsv 

# get sts2015 evaluation set
wget http://alt.qcri.org/semeval2015/task2/data/uploads/sts2015-es-test.zip
mkdir sts2015-es; unzip sts2015-es-test.zip; mv STS.* sts2015-es
```

## Modded sentence-transformers
**DO NOT INSTALL sentence-transformers YET**. Do this instead to be able to
visualize loss in real time and have a file to inspect.

```
git clone https://github.com/UKPLab/sentence-transformers.git
cp SentenceTransformers.py sentence-transformers/sentence_transformers/
```

This will replace the implementation with one that logs the loss value at each
step and it's mean at each epoch. Then just install sentence-transformers from
the local repository

```
pip install -e sentence-transformers
```

## Usage

Use the Model\_Training.ipynb to train the models.
