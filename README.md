![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)&nbsp;![License](https://img.shields.io/github/license/sanjay-kv/Semi-supervised-sequence-learning-Project)&nbsp;![Issues](https://img.shields.io/github/issues/sanjay-kv/Semi-supervised-sequence-learning-Project)&nbsp;![Forks](https://img.shields.io/github/forks/sanjay-kv/Semi-supervised-sequence-learning-Project)

# Adversarial Text Classification

Code for [*Adversarial Training Methods for Semi-Supervised Text Classification*](https://arxiv.org/abs/1605.07725) and [*Semi-Supervised Sequence Learning*](https://arxiv.org/abs/1511.01432).

Source Paper:  [*Semi-Supervised Sequence Learning*]( [Semi-Supervised Sequence Learning*](https://arxiv.org/abs/1511.01432).)

Source Repository:  [*tensorflow/models/research/adversarial_text*]( [https://github.com/tensorflow/models/tree/master/research/adversarial_text)

## Requirements

* TensorFlow = v1.15.5
* Current VM configuration, Current configuration: AWS Ubuntu server 18.04 LTS 64 bit (x86), t2.micro, 30GB SSD.

## End-to-end IMDB Sentiment Classification Replication

### Fetch data

```bash
$ wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz \
    -O /tmp/imdb.tar.gz
$ tar -xf /tmp/imdb.tar.gz -C /tmp
```

The directory `/tmp/aclImdb` contains the raw IMDB data.

### Generate vocabulary

```bash
$ IMDB_DATA_DIR=/tmp/imdb
```

Assigning the folder imdb in tmp to **IMDB_DATA_DIR**

```bash
$ python gen_vocab.py \
    --output_dir=$IMDB_DATA_DIR \
    --dataset=imdb \
    --imdb_input_dir=/tmp/aclImdb \
    --lowercase=False
```

Head to the models/research/adversarial_text  in the cloned repository. Execute the above command. Vocabulary and frequency files will be generated in `$IMDB_DATA_DIR`. the input files will be taken from /tmp/aclImdb 

### Scaling down the file

Head to the tmp/aclimdb filtering the 10% of data for further processing as alternative solution against the memory allocation issue.

```bash
mv neg neg_original
mv pos pos_original
#create a new repository
$mkdir neg
$mkdir pos
mv neg_orginal/9*.txt neg #filtering ~2k files from 25k 
mv pos_orginal/9*.txt pos
#The same need to be performed on the train data, test data , unsup files (unlablled data)
```

###  Generate training, validation, and test data

```bash
$ python gen_data.py \
    --output_dir=$IMDB_DATA_DIR \
    --dataset=imdb \
    --imdb_input_dir=/tmp/aclImdb \
    --lowercase=False \
    --label_gain=False
```

`$IMDB_DATA_DIR` contains TFRecords files.

### Pretrain IMDB Language Model

```bash
$ PRETRAIN_DIR=/tmp/models/imdb_pretrain
$ python pretrain.py \
    --train_dir=$PRETRAIN_DIR \
    --data_dir=$IMDB_DATA_DIR \
    --vocab_size=87007 \
    --embedding_dims=256 \
    --rnn_cell_size=64 \
    --num_candidate_samples=64 \
    --batch_size=1 \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.9999 \
    --max_steps=100000 \
    --max_grad_norm=1.0 \
    --num_timesteps=400 \
    --keep_prob_emb=0.5 \
    --normalize_embeddings
```

`$PRETRAIN_DIR` contains checkpoints of the pretrained language model.

### Train classifier

```bash
$ TRAIN_DIR=/tmp/models/imdb_classify
$ python train_classifier.py \
    --train_dir=$TRAIN_DIR \
    --pretrained_model_dir=$PRETRAIN_DIR \
    --data_dir=$IMDB_DATA_DIR \
    --vocab_size=87007 \
    --embedding_dims=256 \
    --rnn_cell_size=1024 \
    --cl_num_layers=1 \
    --cl_hidden_size=30 \
    --batch_size=64 \
    --learning_rate=0.0005 \
    --learning_rate_decay_factor=0.9998 \
    --max_steps=15000 \
    --max_grad_norm=1.0 \
    --num_timesteps=400 \
    --keep_prob_emb=0.5 \
    --normalize_embeddings \
    --adv_training_method=vat \
    --perturb_norm_length=5.0
```

### Evaluate on test data

```bash
$ EVAL_DIR=/tmp/models/imdb_eval
$ python evaluate.py \
    --eval_dir=$EVAL_DIR \
    --checkpoint_dir=$TRAIN_DIR \
    --eval_data=test \
    --run_once \
    --num_examples=25000 \
    --data_dir=$IMDB_DATA_DIR \
    --vocab_size=87007 \
    --embedding_dims=256 \
    --rnn_cell_size=1024 \
    --batch_size=256 \
    --num_timesteps=400 \
    --normalize_embeddings
```

**Head to** cd /tmp/models/imdb_pretrain, **You get a graph.pbtxt file on the file.**

Copy the folders to the main directory

```bash
cp /tmp/models/imdb_pretrain/* ~/imdb_pretrain
```

### Data Generation

*   Vocabulary generation: [`gen_vocab.py`](https://github.com/tensorflow/models/tree/master/research/adversarial_text/gen_vocab.py)
*   Data generation: [`gen_data.py`](https://github.com/tensorflow/models/tree/master/research/adversarial_text/gen_data.py)

Command-line flags defined in [`document_generators.py`](https://github.com/tensorflow/models/tree/master/research/adversarial_text/data/document_generators.py)
control which dataset is processed and how.

### New Data Generation (IMDB, AMAZON)

*   IMDB Web_Scrapping: [`imdb_scrapping.ipynb`](https://github.com/sanjay-kv/Semi-supervised-sequence-learning-Project/blob/main/imdb_review_scrapping/Movie_review_imdb_scrapping.ipynb)
*   Amazon new data generation: [`amazon_scrapping.py`](https://github.com/sanjay-kv/Semi-supervised-sequence-learning-Project/blob/main/amazon_scrapping/scrapping.py)
*   SVM on Amazon listing: [`amazon_listing.ipynb`](https://github.com/sanjay-kv/Semi-supervised-sequence-learning-Project/blob/main/NLP_Amazon/Amazon_listing.ipynb)

## Replicated by

* @[sanjay-kv](https://github.com/sanjay-kv), @[mona-piya](https://github.com/mona-piya), @[Mohammed-Rizwan-Amanullah](https://github.com/Mohammed-Rizwan-Amanullah), @[Sabihabit](https://github.com/Sabihabit)
* Mail to:  [sanjay](sanjay@recodehive.com), [Mona]( piyakorn.munegan@students.mq.edu.au), [Rizwan]( mohammedrizwan.amanullah@students.mq.edu.au)
* Takeru Miyato, @takerum (Original implementation), Andrew M.Dai adai@google.com (Tensorflow/Models)
