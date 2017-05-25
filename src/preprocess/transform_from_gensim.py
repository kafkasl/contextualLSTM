from gensim.models.keyedvectors import KeyedVectors
from utils.vector_manager import VectorManager
import numpy as np
import sys
import argparse


def transform_gensim(wv):
    print("Transforming from gensim a total of %s" % len(wv.vocab.items()))
    zero_vec = np.zeros_like(wv.syn0[0])
    complete_vec = [(v.index + 1, w, wv.word_vec(w)) for w, v in wv.vocab.items()]
    sorted_vec = sorted(complete_vec)
    id_word_vec = [(0, 'unk', zero_vec)] + sorted_vec
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

    w2id_filepath = "word2id_%s" % emb_size
    idWordVec_filepath = "idWordVec_%s" % emb_size

    print("Writing files:\n\t * word2id: %s\n\t * idWordVec: %s" % (w2id_filepath, idWordVec_filepath))
    VectorManager.write_file(w2id_filepath, word2id)
    VectorManager.write_file(idWordVec_filepath, id_word_vec)