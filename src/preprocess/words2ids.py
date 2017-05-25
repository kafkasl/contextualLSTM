import multiprocessing
import os
import argparse
import multiprocessing as mp
from time import time
import numpy as np

from utils.vector_manager import VectorManager


def word2Id(param):
    filename, w2id = param
    unk_id = 0
    file_out = "%s_num" % filename.split("_clean")[0]

    def transform():
        docs = VectorManager.read_vector(filename)
        # docs = [doc.strip() for doc in data.split(".\n--EOD--\n") if doc.strip()]
        file_list = []
        for doc in docs:
            doc_list = []
            for paragraph in doc:
                par_list = []
                for sentence in paragraph:
                    s_id = [toId(word) for word in sentence if word]
                    if s_id:
                        par_list.append(s_id)
                doc_list.append(par_list)
            file_list.append(doc_list)
        VectorManager.write_file(file_out, file_list)
        np.save(file_out, np.array(file_list))
        return file_list

    def toId(word):
        word_id = unk_id
        # print("W: %s" % word)
        try:
            word_id = w2id[word]
            # print("WordID: %s" % word_id)
        except KeyError:
            pass
        finally:
            return word_id

    return transform()


class FileW2ID(object):

    def __init__(self, filepaths, w2id):
        self.filepaths = filepaths
        self.w2id = w2id

    def __iter__(self):
        for file in self.filepaths:
            yield (file, self.w2id)


def translate_files(data_path, w2id):
    print("Translating files from %s" % (data_path))
    print("Creating multiprocessing pool.")
    # p = mp.Pool(multiprocessing.cpu_count() * 2)

    filepaths = []
    for root, dirs, files in os.walk(data_path):
        filepaths.extend(["%s/%s" % (root, file) for file in files if file.endswith("_clean.pklz")])

    print("Starting word2Ids with %s processes and %s files" %
          (multiprocessing.cpu_count() * 2, len(filepaths)))
    iter_file_w2id = FileW2ID(filepaths, w2id)

    for p in iter_file_w2id:
        word2Id(p)
    # p.map(word2Id, iter_file_w2id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help="Path of the data to be translated with word2id vector."
                                                       " and clean up.", required=True)
    parser.add_argument('-wv', '--word_vector', type=str, help="Word2ID vector to be used for doc translation.",
                        required=True)

    args = parser.parse_args()
    data_path = args.data
    word2id_file = args.word_vector

    begin = time()

    w2Id = VectorManager.read_vector(word2id_file)
    translate_files(data_path, w2Id)

    end = time()
    print("Total procesing time: %d seconds" % (end - begin))
