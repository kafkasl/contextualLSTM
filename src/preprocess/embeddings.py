from utils.vector_manager import VectorManager
from time import time

import multiprocessing as mp
import gensim
import os
import sys
import argparse
from contextlib import closing



def read_file(filename):
    return VectorManager.read_vector(filename)


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.files = []
        self.file_paths = []
        for root, dirs, files in os.walk(self.dirname):
            for filename in [file for file in files if file.endswith("_simple")]:
                file_path = root + '/' + filename
                self.file_paths.append(file_path)
        print("Got %s files to turn into sentences" % len(self.file_paths))

        # threads = mp.cpu_count()
        # with closing(mp.Pool(threads, maxtasksperchild=10)) as p:
        #
        #     file_vecs = p.map(read_file, file_paths)
        #     self.files.append(file_vecs)
        #     print("My Sentences initialized with all data in memory.")

    def __iter__(self):
        """
        Defines how to iterate the MySentences class in order to feed it directly into Word2Vec method. Yields a
        sentence (as a list of words) for every iteration.
        """
        # for root, dirs, files in os.walk(self.dirname):
        for file_path in self.file_paths:
            file_data = VectorManager.read_vector(file_path)
            file_sentences = VectorManager.parse_into_sentences(file_data)

            for sentence in file_sentences:
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
    print("[BLOCK] Initializing MySentences")
    sentences = MySentences(files_path)
    print("[BLOCK] Creating embeddings model")
    sys.stdout.flush()
    model_w2v = gensim.models.Word2Vec(sentences,
                                       size=embedding_size,
                                       window=10,
                                       min_count=minimum_count,
                                       workers=mp.cpu_count())
    print("[BLOCK] Created embeddings of size %s" % embedding_size)
    sys.stdout.flush()

    return model_w2v


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help="Path of the data to be used for the word embeddings"
                                                       " and clean up.", required=True)
    parser.add_argument('-s', '--size', type=int, help="Size of the word embeddings.", default=200, required=True)
    parser.add_argument('-c', '--min_count', type=int, help="Size of the word embeddings.", default=200, required=False)

    args = parser.parse_args()
    data_path = args.data
    emb_size = args.size
    min_count = args.min_count

    print("Creating embeddings of size %s for data in %s" % (emb_size, data_path))

    begin = time()

    model = create_embeddings(data_path, emb_size, min_count)

    print("Saving embeddings model...")
    model.save("../models/word2vec_gensim_%s" % emb_size)
    model.wv.save_word2vec_format("../models/word2vec_org_%s" % emb_size,
                                  "../models/vocabulary_%s" % emb_size,
                                  binary=False)

    end = time()
    print("Total processing time: %d seconds" % (end - begin))
