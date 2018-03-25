
python ../src/lstm/lstm.py \
        --data_path ../data/full.list \
        --embeddings ../models/eos/idWordVec_ \
        --model large \
        --use_fp16 True \
        --word_to_id ../models/eos/word2id_1000.pklz

