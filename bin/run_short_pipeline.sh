#!/bin/bash

export PYTHONPATH="$PYTHONPATH:../src/"

data_path=$1
embeddings_size=$2


python2 ../src/preprocess/filter.py \
    --data ${data_path} \
    --word_vector ../models/word2id_1000.pklz

# Translate all wiki data
python2 ../src/preprocess/words2ids.py \
    --data ${data_path} \
    --word_vector ../models/eos/word2id_1000.pklz


# Put all the files into a list to be fed to TF LSTM
find ../data/ -name *num_eos > ../data/full.list

# Run the LSTM 
python ../src/lstm/lstm.py \
        --data_path ../data/full.list \
        --embeddings ../models/idWordVec_${embeddings_size}.pklz \
        --model large \
        --use_fp16 True \
        --word_to_id ../models/word2id_${embeddings_size}.pklz
