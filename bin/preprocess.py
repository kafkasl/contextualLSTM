import sys
sys.path.insert(0, "../src/")

from preprocess.cleaner import clean_data
from preprocess.embeddings import create_embeddings
from preprocess.transform_from_gensim import transform_gensim
from preprocess.words2ids import translate_files
from preprocess.words2ids_validator import check_translated_files
from utils.vector_manager import VectorManager
from time import time

import argparse

""""
Orchestrating file which handles all the processing pipeline:
* Clean data
* Create embeddings
* Transform model structures
* Translate word's lists to IDs lists.
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help="Path of the data extracted with wikiExtractor", required=True)
    parser.add_argument('-s', '--size', type=int, help="Size of the word embeddings.", default=200, required=False)
    parser.add_argument('-c', '--count', type=int, help="Min count for embeddings.", default=200, required=False)

    args = parser.parse_args()

    # Arguments parsing
    data_path = args.data
    emb_size = args.size  # size of the embedding vectors to create
    min_count = args.count  # minimum word occurrences to be in embeddings set

    print("Starting Preprocess pipeline\n\t * Data path: %s\n\t * Embedding size: %s\n\t * Min count: %s" %
          (data_path, emb_size, min_count))

    # Clean Wikipedia data
    t0 = time()
    sentences = clean_data(data_path)
    t1 = time()
    print("Time cleaning data: %s\nCreating embeddings from cleaned data..." % (t1-t0))

    # Create embeddings from the cleaned data
    model = create_embeddings(data_path, emb_size, min_count)
    t2 = time()
    print("Time creating embeddings: %s" % (t2-t1))

    print("Saving embeddings model...")
    model.save("../models/word2vec_gensim_%s" % emb_size)
    model.wv.save_word2vec_format("../models/word2vec_org_%s" % emb_size,
                                  "../models/vocabulary_%s" % emb_size,
                                  binary=False)

    # Get only:
    #  * word2id vector (for transforming data to numerical)
    #  * id_word_vec (actually contain word embeddings an associated id <-> word
    t3 = time()
    word2id, id_word_vec = transform_gensim(model.wv)
    t4 = time()
    print("Time transforming gensim to word2ID and idWordVec vectors: %s" % (t4-t3))

    # Save model for checkpointing
    VectorManager.write_pickled("../models/word2id_%s" % emb_size, word2id)
    VectorManager.write_pickled("../models/idWordVec_%s" % emb_size, id_word_vec)

    t5 = time()
    translate_files(data_path, word2id)
    t6 = time()
    print("Time translating words to numbers: %s" % (t6-t5))

    t7 = time()
    check_translated_files(data_path, word2id)
    t8 = time()
    print("Time translating words to numbers: %s" % (t8-t7))







