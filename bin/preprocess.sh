#!/bin/bash 

export PYTHONPATH="$PYTHONPATH:../src/"

data_path=$1
embeddings_size=$2
min_word_count_threshold=$3

# Preprocess all wiki data and create embeddings
python2 preprocess.py \
    --data ${data_path} \
    --size ${embeddings_size} \


# Put all the files into a list to be fed to TF LSTM
find ../data/ -name *num_eos > ../data/full.list
