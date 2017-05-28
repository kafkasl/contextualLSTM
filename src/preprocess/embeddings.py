from utils.vector_manager import VectorManager
from pattern.en import tokenize
from time import time

import multiprocessing as mp
import gensim
import os
import re
import sys
import argparse


def cleanhtml(raw_html):
    """
    Removes the <doc> tags remaining from wikiExtracted data
    :param raw_html: html/text content of a file with many docs
    :return: only text from raw_html
    """
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext


def remove_title(text):
    """
    Removes the title of a document
    :param text: text containing an article output from cleanhtml()
    :return: text of the article without title
    """
    index = text.find("\n\n")
    if index != -1:
        return text[index+2:]
    else:
        return text


def is_number(s):
    """
    Checks if the parameter s is a number
    :param s: anything
    :return: true if s is a number, false otherwise
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def _transform_file(file_path):
    """
    Transforms a file containing articles into a 4D list of words divided into sentences,
    paragraphs and docs. Write the result to disk with the name filename_clean.pklz
    :param file_path: file to transform
    """
    print("Cleaning %s" % file_path)
    with open(file_path) as f:
        data = f.read()
        docs = data.split("</doc>")
        del data
    file_out = "%s_clean" % file_path
    file_list = []
    for doc in [d.strip() for d in docs if d.strip()]:
        doc_list = []
        paragraphs = [tokenize(par.decode("latin-1")) for par in remove_title(cleanhtml(doc)).strip().split("\n\n")]

        for p in paragraphs:
            par_list = []
            for sent in p:
                line = [word for word in sent.lower().split()
                        if word.isalpha() or is_number(word)]
                if line:
                    par_list.append(line)
            doc_list.append(par_list)
        file_list.append(doc_list)
    VectorManager.write_file(file_out, file_list)
    del file_list
    print("Done with %s" % file_path)


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def transform(self):
        """
        Handles the parallel transformation of all the dataset into 4D lists
        """
        for root, dirs, files in os.walk(self.dirname):
            filtered_files = ["%s/%s" % (root, file) for file in files if is_number(file.split("_")[1]) and len(file.split("_")) == 2]

            threads = mp.cpu_count()*4
            print("Starting %s processes to clean %s files" % (threads, len(filtered_files)))
            i = 0
            while i < len(filtered_files):
                ps = []
                j = 0
                while j < threads and (i+j) < len(filtered_files):
                    print("[%s] Creating %s of %s for file %s" % (i, i+j, max(threads, len(filtered_files)), filtered_files[i+j]))
                    p = (mp.Process(target=_transform_file, args=(filtered_files[i+j],)))
                    p.start()
                    ps.append(p)
                    j += 1

                print("%s process in the list to join" % len(ps))
                j = 0
                while j < threads and (i+j) < len(filtered_files):
                    print("[%s] Joining %s of %s for file %s" % (i, j, max(threads, len(filtered_files)), filtered_files[i+j]))
                    ps[j].join()
                    j += 1

                i += j

        sys.stdout.flush()

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
                            is_alpha_word_line = [word for word in
                                                  sentence
                                                  if word.isalpha() or is_number(word)]
                            if is_alpha_word_line:
                                yield is_alpha_word_line


def clean_data(data_path):

    """
    Wrapper function to cleans the data and transforms it into 4D. Used to be called from either main or as block of
     the pipeline
    :param data_path: of the files to convert
    :return: MySentences class ready to be fed to Word2Vec model
    """
    sentences = MySentences(data_path)
    print("[BLOCK] Transforming sentences to 4-dimensional lists")
    sentences.transform()
    print("[BLOCK] Done transforming data")
    sys.stdout.flush()

    return sentences


def create_embeddings(sentences, emb_size, min_count):

    """
    Creates embeddings with the sentences, embedding size, min_count of occurrences, a max window length of 10, and
    cpu_count() number of workers. Used to be called from either main or as block of the pipeline
    :param sentences: used to generate the word embeddings
    :param emb_size: size of the embeddings to generate
    :param min_count: min. occurrences per word to be included
    :return: word2vec model with all the embeddings and extra info
    """
    print("[BLOCK] Creating embeddings model")
    model = gensim.models.Word2Vec(sentences,
                                   size=emb_size,
                                   window=10,
                                   min_count=min_count,
                                   workers=mp.cpu_count())
    print("[BLOCK] Created embeddings of size %s" % emb_size)
    sys.stdout.flush()

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help="Path of the data to be used for the word embeddings"
                                                       " and clean up.", required=True)
    parser.add_argument('-s', '--size', type=int, help="Size of the word embeddings.", default=200, required=True)

    args = parser.parse_args()
    data_path = args.data
    emb_size = args.size

    print("Parsing data from %s\nEmbedding size: %s" % (data_path, emb_size))

    begin = time()

    sentences = clean_data(data_path)

    min_count = 200

    model = create_embeddings(sentences, emb_size, min_count)

    print("Saving embeddings model...")
    model.save("../models/word2vec_gensim_%s" % emb_size)
    model.wv.save_word2vec_format("../models/word2vec_org_%s" % emb_size,
                                  "../models/vocabulary_%s" % emb_size,
                                  binary=False)

    end = time()
    print("Total processing time: %d seconds" % (end - begin))
