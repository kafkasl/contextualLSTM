import numpy as np
import pickle


# Vectors represent: word <=> id/index <=> embedding
# Auxiliar class handling all the read/write operations for data structures other than numpy arrays.
class VectorManager(object):

    # Methods used to save the vectors
    @staticmethod
    def parse_into_4D(file_string):
        return [[[[w for w in s.split() if w]
                for s in p.split("\n") if s]
                for p in doc.split("\n\n") if p]
                for doc in file_string.split("\n\n\n")
                if doc]

    # Methods used to save the vectors
    @staticmethod
    def parse_into_list(file_string):
        file_list = []
        for doc in file_string.split("\n\n\n"):
            for p in doc.split("\n\n"):
                for s in p.split("\n"):
                    for w in s.split():
                        if w:
                            file_list.append(w)

        return file_list

    # Methods used to save the vectors
    @staticmethod
    def parse_into_sentences(file_string):
        sentences = []
        for doc in file_string.split("\n\n\n"):
            for p in doc.split("\n\n"):
                for s in p.split("\n"):
                    ws = s.split()
                    if ws:
                        sentences.append(ws)
        return sentences

    # Methods used to save the vectors
    @staticmethod
    def write_pickled(filename, data):
        with open('%s.pklz' % filename, 'wb') as f:
            pickle.dump(data, f)

    # Methods used to save the vectors
    @staticmethod
    def write_string(filename, data):
        with open('%s' % filename, 'wb') as f:
            f.write(data)

    # Methods to read vectors
    @staticmethod
    def read_vector(filename):
        ext = filename.split(".")[-1]

        if ext == "npy":
            with open(filename, "rb") as f:
                return np.load(f)
        if ext == "pklz":
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            with open(filename, 'rb') as f:
                data = f.read()
                return data.decode("latin-1")
                # return data.decode("latin-1")
            # print("Unknown file extension for file %s" % filename)

    @staticmethod
    def read_id_word_vec():
        return self.read_vector("idWordVec.pklz")

    @staticmethod
    def read_word2id():
        return self.read_vector("word2id.pklz")