#!/bin/bash 

export PYTHONPATH="$PYTHONPATH:../src/"

data_path=$1
embeddings_size=$2

# Preprocess all wiki data and create embeddings
python2 preprocess.py \
    --data ${data_path} \
    --size ${embeddings_size} \


# Put all the files into a list to be fed to TF LSTM
find ../data/ -name *num_eos > ../data/full.list

# Run the LSTM 
python ../src/lstm/lstm.py \
        --data_path ../data/full.list \
        --embeddings ../models/eos/idWordVec_ \
        --model large \
        --use_fp16 True \
        --word_to_id ../models/eos/word2id_${embeddings_size}.pklz
