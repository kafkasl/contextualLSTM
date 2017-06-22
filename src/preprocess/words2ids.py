from utils.vector_manager import VectorManager
from time import time
import multiprocessing as mp
import argparse
import numpy as np
import os
import sys


def word2Id(filename, w2id, debug=False):
    if debug:
        print("Translating %s" % filename)
    unk_id = 0
    file_out = "%s_num" % filename.split("_clean")[0]

    def transform_numpy():
        """
        Transforms a 4D list of words into a 4D numpy array of integers and writes it into file_out
        """
        docs = VectorManager.parse_into_4D(VectorManager.read_vector(filename))
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


    def transform():
        """
        Transforms a 4D list of words into a 4D numpy array of integers and writes it into file_out
        """
        with open(filename) as f:
            data = f.read().decode("latin-1").split()

        ids = " ".join([str(w2id[w]) for w in data])

        with open("%s_num_eos" % filename, "wb") as f:
            f.write(ids)


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

    transform()
    # return transform()


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


def translate_files(data_path, w2id, suffix="_simple", debug=False):
    """
    Handles the parallel translation from word to id of the files in data_path with the mapping w2id
    :param data_path: path of the files to transform. Used to be called from either main or as block of
     the pipeline
    :param w2id: mappings to be used
    """
    print("[BLOCK] Translating files from %s" % (data_path))

    filepaths = []
    for root, dirs, files in os.walk(data_path):
        filepaths.extend(["%s/%s" % (root, file) for file in files if file.endswith(suffix)])

    threads = min(mp.cpu_count() * 4, filepaths)

    print("[BLOCK] Starting %s processes to translate to IDs %s files" % (threads, len(filepaths)))
    i = 0
    while i < len(filepaths):
        ps = []
        j = 0
        while j < threads and (i + j) < len(filepaths):
            if debug:
                print("[%s] Creating %s of %s for file %s" % (
                    i, i + j, len(filepaths), filepaths[i + j]))
            p = (mp.Process(target=word2Id, args=(filepaths[i + j], w2id,)))
            p.start()
            ps.append(p)
            j += 1

        if debug:
            print("%s process in the list to join" % len(ps))
        j = 0
        while j < threads and (i + j) < len(filepaths):
            if debug:
                print("[%s] Joining %s of %s for file %s" % (
                    i, j, len(filepaths), filepaths[i + j]))
            ps[j].join()
            j += 1

        i += j
    # for p in iter_file_w2id:
    #     word2Id(p)
    # p = mp.Pool(threads, maxtasksperchild=1)
    # p.map(word2Id, iter_file_w2id)

    print("[BLOCK] Files translated to IDs")
    sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help="Path of the data to be translated with word2id vector."
                                                       " and clean up.", required=True)
    parser.add_argument('-w', '--word_vector', type=str, help="Word2ID vector to be used for doc translation.",
                        required=False, default="../models/eos/word2id_1000.pklz")

    args = parser.parse_args()
    data_path = args.data
    word2id_file = args.word_vector

    begin = time()

    w2Id = VectorManager.read_vector(word2id_file)
    translate_files(data_path, w2Id)

    end = time()
    print("Total processing time: %d seconds" % (end - begin))
