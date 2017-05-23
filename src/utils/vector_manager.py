import gzip
import pickle


# Vectors represent: word <=> id/index <=> embedding
class VectorManager(object):

    # Methods used to save the vectors
    @staticmethod
    def write_file(filename, data):
        with gzip.open('%s.pklz' % filename, 'wb') as f:
            pickle.dump(data, f)

    # Methods to read vectors
    @staticmethod
    def read_vector(filename):
        with gzip.open(filename, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def read_id_word_vec():
        return self.read_vector("idWordVec.pklz")

    @staticmethod
    def read_word2id():
        return self.read_vector("word2id.pklz")