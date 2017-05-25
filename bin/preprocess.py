import sys
sys.path.insert(0, '../src/')

from preprocess.embeddings import clean_data, create_embeddings
from preprocess.transform_from_gensim import transform_gensim
from preprocess.words2ids import translate_files
from utils.vector_manager import VectorManager


import argparse



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

    # Clean Wikipedia data
    sentences = clean_data(data_path)

    # Create embeddings from the cleaned data
    model = create_embeddings(sentences, emb_size, min_count)

    print("Saving embeddings model...")
    model.save("../models/word2vec_gensim_%s" % emb_size)
    model.wv.save_word2vec_format("../models/word2vec_org_%s" % emb_size,
                                  "../models/vocabulary_%s" % emb_size,
                                  binary=False)

    # Get only:
    #  * word2id vector (for transforming data to numerical)
    #  * id_word_vec (actually contain word embeddings an associated id <-> word
    word2id, id_word_vec = transform_gensim(model.wv)

    # Save model for checkpointing
    VectorManager.write_file("../models/word2id_%s" % emb_size, word2id)
    VectorManager.write_file("../models/idWordVec_%s" % emb_size, id_word_vec)

    translate_files(data_path, word2id)







