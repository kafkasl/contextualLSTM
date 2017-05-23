from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from utils.vector_manager import VectorManager
import sys


def transform_gensim(wv):
    zero_vec = np.zeros_like(wv.syn0[0])
    complete_vec = [(v.index + 1, w, wv.word_vec(w)) for w, v in wv.vocab.items()]
    sorted_vec = sorted(complete_vec)
    id_word_vec = [(0, 'unk', zero_vec)] + sorted_vec
    word2id = [(w, id) for id, w, _ in id_word_vec]

    return word2id, id_word_vec

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Please use python transform_from_gensim.py keyed_vectors_path")
        exit()

    data_path = sys.argv[1]

    wv = KeyedVectors.load_word2vec_format(data_path, binary=False)

    emb_size = len(wv.syn0[0])
    word2id, id_word_vec = transform_gensim(wv)

    VectorManager.write_file("word2id_%s" % emb_size, word2id)
    VectorManager.write_file("idWordVec_%s" % emb_size, id_word_vec)