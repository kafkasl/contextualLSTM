# contextualLSTM
Contextual LSTM for NLP tasks like word prediction
 
This repo's goal is to implement de Contextual LSTM model for word prediction as described by [Ghosh, S., Vinyals, O., Strope, B., Roy, S., Dean, T., & Heck, L. (n.d.). Contextual LSTM (CLSTM) models for Large scale NLP tasks. https://doi.org/10.1145/12351]


## Data preprocessing and embeddings

Further details about wikipedia data preprocessing at

./documentation/word_embeddings_and_topic_detection.pdf


## Context creation with topic detection

Further details of different gensim topic detection methods as well as embeddings arithmetic for context creation at

./documentation/word_embeddings_and_topic_detection_II.pdf

## Execution

Download a wikipedia dump for example:

http://ftp.acc.umu.se/mirror/wikimedia.org/dumps/enwiki/20180120/enwiki-20180120-pages-articles.xml.bz2

After that use wiki_extractor to process it:

`./wiki_extractor_launch.sh path_to_wikipedia_dump`

where `path_to_wikipedia_dump` is the file you downloaded (e.g. enwiki-20180120-pages-articles.xml.bz2)

Next step is to filter the data with the pretrained embeddings:

`python preprocess/filter.py --data ../data/enwiki/ -w ../models//word2id_1000.pklz`

Next step is to preprocess the wikipedia data to create the embeddings and support structures:


`./preprocess.sh ../data/enwiki 500 20`

where:
 * `../data/enwiki` is the default path where preprocess script extracted and cleaned the wikipedia dump.
 * 500 is the desired embedding size.
 * 20 is the minimum number of times a word has to occur to be included in the embeddings.


Finally run LSTM with:

```
python ../src/lstm/lstm.py \
        --data_path ../data/wikipedia/full.list \
        --embeddings ../models/eos/idWordVec_ \
        --model large \
        --use_fp16 True \
        --word_to_id ../models/eos/word2id_500.pklz
```
## LSTM 

Basic LSTM implementation with TF at  ./src/lstm_frag.py

## CLSTM 

Contextual LSTM implementation with TF at  ./src/clstm.py

**Although functional, this version is still too slow to be practical for training. If you want to collaborate or have any question regarding it feel free to contact me, I plan to finish it shortly and upload a detailed description of it.**


## Execution

Most files have their own execution script under /bin folder.
All scripts named submit_XXX.sh are designed to be run in a SuperComputer with Slurm queue system. In order to run the locally, just issue the python commands followed by the correct paths. **Note:** due to the use of many different packages not all files run with the same Python version (some with 2.7, others with 3.5.2 and the rest 3.6), I expect to unify them (or state clearly the version) soon.
