import numpy as np
import gzip
import pickle


# Vectors represent: word <=> id/index <=> embedding
# Auxiliar class handling all the read/write operations for data structures other than numpy arrays.
class VectorManager(object):

    # Methods used to save the vectors
    @staticmethod
    def write_file(filename, data):
        with gzip.open('%s.pklz' % filename, 'wb') as f:
            pickle.dump(data, f)

    # Methods to read vectors
    @staticmethod
    def read_vector(filename):
        ext = filename.split(".")[-1]
        if ext == "pklz":
            with gzip.open(filename, 'rb') as f:
                return pickle.load(f)
        elif ext == "npy":
            with open(filename, "rb") as f:
                return np.load(f)
        else:
            print("Unknown file extension for file %s" % filename)

    @staticmethod
    def read_id_word_vec():
        return self.read_vector("idWordVec.pklz")

    @staticmethod
    def read_word2id():
        return self.read_vector("word2id.pklz")