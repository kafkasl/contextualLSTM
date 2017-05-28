from utils.vector_manager import VectorManager
from time import time
import multiprocessing as mp
import argparse
import numpy as np
import os
import sys



def word2Id(param):
    filename, w2id = param
    print("Translating %s" % filename)
    unk_id = 0
    file_out = "%s_num" % filename.split("_clean")[0]

    def transform():
        """
        Transforms a 4D list of words into a 4D numpy array of integers and writes it into file_out
        """
        docs = VectorManager.read_vector(filename)
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
        np.save(file_out, np.array(file_list))
        return np.array(file_list)

    def toId(word):
        """
        Return ID of the word (or 0 if word is not in word2Id dict)
        :param word: to translated
        :return: Id of the word
        """
        word_id = unk_id
        try:
            word_id = w2id[word]
        except KeyError:
            pass
        finally:
            return word_id

    return transform()


class FileW2ID(object):
    """
    Auxiliar class which holds the filepaths and w2id structure and yields them one at a time in order to avoid
    replicating the w2id structure (which can be quite big)
    """

    def __init__(self, filepaths, w2id):
        self.filepaths = filepaths
        self.w2id = w2id

    def __iter__(self):
        for file in self.filepaths:
            yield (file, self.w2id)


def translate_files(data_path, w2id):
    """
    Handles the parallel translation from word to id of the files in data_path with the mapping w2id
    :param data_path: path of the files to transform. Used to be called from either main or as block of
     the pipeline
    :param w2id: mappings to be used
    """
    print("[BLOCK] Translating files from %s" % (data_path))

    filepaths = []
    for root, dirs, files in os.walk(data_path):
        filepaths.extend(["%s/%s" % (root, file) for file in files if file.endswith("_clean.pklz")])

    print("[BLOCK] Starting word2Ids with %s processes and %s files" %
          (mp.cpu_count() * 2, len(filepaths)))
    iter_file_w2id = FileW2ID(filepaths, w2id)

    p = mp.Pool(mp.cpu_count() * 2)
    p.map(word2Id, iter_file_w2id)

    print("[BLOCK] Files translated to IDs")
    sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help="Path of the data to be translated with word2id vector."
                                                       " and clean up.", required=True)
    parser.add_argument('-w', '--word_vector', type=str, help="Word2ID vector to be used for doc translation.",
                        required=True)

    args = parser.parse_args()
    data_path = args.data
    word2id_file = args.word_vector

    begin = time()

    w2Id = VectorManager.read_vector(word2id_file)
    translate_files(data_path, w2Id)

    end = time()
    print("Total processing time: %d seconds" % (end - begin))
