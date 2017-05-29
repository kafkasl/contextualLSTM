from utils.vector_manager import VectorManager
from time import time

import multiprocessing as mp
import gensim
import os
import sys
import argparse


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        """
        Defines how to iterate the MySentences class in order to feed it directly into Word2Vec method. Yields a
        sentence (as a list of words) for every iteration.
        """
        for root, dirs, files in os.walk(self.dirname):
            for filename in [file for file in files if file.endswith("_clean.pklz")]:
                file_path = root + '/' + filename
                file_vecs = VectorManager.read_vector(file_path)

                for doc in file_vecs:
                    for p in doc:
                        for sentence in p:
                            if sentence:
                                yield sentence


def create_embeddings(files_path, embedding_size, minimum_count):

    """
    Creates embeddings with the sentences, embedding size, min_count of occurrences, a max window length of 10, and
    cpu_count() number of workers. Used to be called from either main or as block of the pipeline
    :param files_path: used to generate the word embeddings
    :param embedding_size: size of the embeddings to generate
    :param minimum_count: min. occurrences per word to be included
    :return: word2vec model with all the embeddings and extra info
    """
    sentences = MySentences(files_path)
    print("[BLOCK] Creating embeddings model")
    model_w2v = gensim.models.Word2Vec(sentences,
                                       size=embedding_size,
                                       window=10,
                                       min_count=minimum_count,
                                       workers=mp.cpu_count())
    print("[BLOCK] Created embeddings of size %s" % emb_size)
    sys.stdout.flush()

    return model_w2v


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help="Path of the data to be used for the word embeddings"
                                                       " and clean up.", required=True)
    parser.add_argument('-s', '--size', type=int, help="Size of the word embeddings.", default=200, required=True)

    args = parser.parse_args()
    data_path = args.data
    emb_size = args.size

    print("Creating embeddings of size %s for data in %s" % (emb_size, data_path))

    begin = time()

    min_count = 200

    model = create_embeddings(data_path, emb_size, min_count)

    print("Saving embeddings model...")
    model.save("../models/word2vec_gensim_%s" % emb_size)
    model.wv.save_word2vec_format("../models/word2vec_org_%s" % emb_size,
                                  "../models/vocabulary_%s" % emb_size,
                                  binary=False)

    end = time()
    print("Total processing time: %d seconds" % (end - begin))
