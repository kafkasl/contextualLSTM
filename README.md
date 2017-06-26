# contextualLSTM
Contextual LSTM for NLP tasks like word prediction
 
This repo's goal is to implement de Contextual LSTM model for word prediction as described by [Ghosh, S., Vinyals, O., Strope, B., Roy, S., Dean, T., & Heck, L. (n.d.). Contextual LSTM (CLSTM) models for Large scale NLP tasks. https://doi.org/10.1145/12351]


## Data preprocessing and embeddings

Further details about wikipedia data preprocessing at

./documentation/word_embeddings_and_topic_detection.pdf


## Context creation with topic detection

Further details of different gensim topic detection methods as well as embeddings arithmetic for context creation at

./documentation/word_embeddings_and_topic_detection_II.pdf

## LSTM 

Basic LSTM implementation with TF at  ./src/lstm_frag.py

## CLSTM 

Contextual LSTM implementation with TF at  ./src/clstm.py

**Although functional, this version is still too slow to be practical for training. If you want to collaborate or have any question regarding it feel free to contact me, I plan to finish it shortly and upload a detailed description of it.**
