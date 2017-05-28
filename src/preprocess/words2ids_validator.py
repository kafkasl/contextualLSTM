from utils.vector_manager import VectorManager
from time import time
import multiprocessing as mp
import argparse
import numpy as np
import os
import sys

confidence = 0.8


def id2Word(param):
    filename, id2w = param
    file_words = "%s_clean.pklz" % filename.split("_num")[0]
    print("Comparing original %s with %s" % (file_words, filename))


    def is_valid():
        """
        Transforms a 4D list of words into a 4D numpy array of integers and writes it into file_out
        """
        docs_ids = VectorManager.read_vector(filename)
        original = VectorManager.read_vector(file_words)
        file_list = []
        comparison = []
        for d in range(0, len(docs_ids)):
            doc_list = []
            for p in range(0, len(docs_ids[d])):
                par_list = []
                for s in range(0, len(docs_ids[d][p])):
                    sent_list = []
                    for w in range(0, len(docs_ids[d][p][s])):
                        translated = to_word(docs_ids[d][p][s][w])
                        comparison.append(translated == original[d][p][s][w])
                        sent_list.append(translated)
                    par_list.append(sent_list)
                doc_list.append(par_list)
            file_list.append(doc_list)

        valid = False
        try:
            ratio = float(comparison.count(True)) / len(comparison)
            if ratio < confidence:
                print("[WARN] File %s equality ratio is %s" % (filename, round(ratio, 2)))
            else:
                print("[OK] File %s equality ratio is %s" % (filename, round(ratio, 2)))
                valid = True
        except KeyError as e:
            print("[ERROR] File %s is completely different (%s)" % (filename, e))


        return valid


    def to_word(id):
        """
        Return Word associated with id
        :param id: of the word to translate
        :return: word associated with the ID
        """
        try:
            word = id2w[id]
        except IndexError as e:
            print("ID %s not found\n%s" % (id, e))
            word = 'unk'
        return word

    return is_valid()


class FileID2Word(object):
    """
    Auxiliar class which holds the filepaths and w2id structure and yields them one at a time in order to avoid
    replicating the w2id structure (which can be quite big)
    """

    def __init__(self, filepaths, id2w):
        self.filepaths = filepaths
        self.id2w = id2w

    def __iter__(self):
        for file in self.filepaths:
            yield (file, self.id2w)


def check_translated_files(data_path, w2Id):
    """
    Handles the parallel translation from word to id of the files in data_path with the mapping w2id
    :param data_path: path of the files to transform. Used to be called from either main or as block of
     the pipeline
    :param w2id: mappings to be used
    """
    print("[BLOCK] Validating translated files from %s" % (data_path))

    sorted_list = sorted(w2Id.items(), key= lambda(x): x[1])
    id2words = [w for w,_ in sorted_list]
    del w2Id, sorted_list
    filepaths = []
    for root, dirs, files in os.walk(data_path):
        filepaths.extend(["%s/%s" % (root, file) for file in files if file.endswith("_num.npy")])
    n_threads = mp.cpu_count() * 2
    print("[BLOCK] Starting validation with %s processes and %s files" % (n_threads, len(filepaths)))
    iter_file_w2id = FileID2Word(filepaths, id2words)

    p = mp.Pool(n_threads)
    valids = p.map(id2Word, iter_file_w2id)
    print("[BLOCK] Validation done. Correct files %s/%s. Confidence [%s]" % (valids.count(True), len(valids), confidence))
    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help="Path of the data to be translated with word2id vector."
                                                       " and clean up.", required=True)
    parser.add_argument('-w ', '--word_vector', type=str, help="Word2ID vector to be used for doc reverse translation.",
                        required=True)

    args = parser.parse_args()
    data_path = args.data
    word2id_file = args.word_vector

    begin = time()

    w2Id = VectorManager.read_vector(word2id_file)
    check_translated_files(data_path, w2Id)

    end = time()
    print("Total processing time: %d seconds" % (end - begin))
