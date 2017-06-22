from gensim.models.keyedvectors import KeyedVectors
from utils.vector_manager import VectorManager
import numpy as np
import argparse


def transform_gensim(wv):
    """
    Transforms word2Vec model class to two structures: word2id dictionary (used to translate word into IDs) and
    id_word_vec which contains the tuple (id, word, embedding) for each word in the model. Used to be called from
    either main or as block of the pipeline.
    :param wv: word2vec model with the word embeddings
    :return: word2id and id_word_vec
    """
    print("Transforming from gensim a total of %s" % len(wv.vocab.items()))
    complete_vec = [(v.index, w, wv.word_vec(w)) for w, v in wv.vocab.items()]
    sorted_vec = sorted(complete_vec)
    id_word_vec = sorted_vec
    word2id = dict([(w, id) for id, w, _ in id_word_vec])

    return word2id, id_word_vec

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kv', type=str, help="Path of the keyed vectors to translate [word2vec_org_XXX]",
                        required=True)

    args = parser.parse_args()
    data_path = args.kv

    print("Loading keyed vectors")
    wv = KeyedVectors.load_word2vec_format(data_path, binary=False)

    emb_size = len(wv.syn0[0])
    word2id, id_word_vec = transform_gensim(wv)

    w2id_filepath = "../models/word2id_%s" % emb_size
    idWordVec_filepath = "../models/idWordVec_%s" % emb_size

    print("Writing files:\n\t * word2id: %s\n\t * idWordVec: %s" % (w2id_filepath, idWordVec_filepath))
    VectorManager.write_pickled(w2id_filepath, word2id)
    VectorManager.write_pickled(idWordVec_filepath, id_word_vec)