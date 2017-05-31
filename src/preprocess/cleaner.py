from utils.vector_manager import VectorManager
from pattern.en import tokenize
from time import time

import multiprocessing as mp
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


def _transform_file(file_path, debug=False):
    """
    Transforms a file containing articles into a 4D list of words divided into sentences,
    paragraphs and docs. Write the result to disk with the name filename_clean.pklz
    :param file_path: file to transform
    """
    if debug:
        print("Cleaning %s" % file_path)
    with open(file_path) as f:
        data = f.read().decode("latin-1")
        docs = data.split("</doc>")
        del data
    file_out = "%s_clean" % file_path
    file_string = ""
    for doc in [d.strip() for d in docs if d.strip()]:
        paragraphs = [tokenize(par) for par in remove_title(cleanhtml(doc)).strip().split("\n\n") if par]
        doc_a = False
        for p in paragraphs:
            par_a = False
            for sent in p:
                line = " ".join([word for word in sent.lower().split()
                                 if word.isalpha() or is_number(word)])
                if line:
                    file_string += line + "\n"
                    par_a = True
                    doc_a = True

            if par_a:
                file_string += "\n"
        if doc_a:
            file_string += "\n"

    VectorManager.write_string(file_out, file_string.encode("latin-1"))
    del file_string
    if debug:
        print("Done with %s" % file_path)


def transform(dirname, debug=False):
    """
    Handles the parallel transformation of all the dataset into 4D lists
    """
    for root, dirs, files in os.walk(dirname):
        filtered_files = ["%s/%s" % (root, file) for file in files if
                          is_number(file.split("_")[1]) and len(file.split("_")) == 2]

        threads = min(mp.cpu_count() * 4, filtered_files)
        print("Starting %s processes to clean %s files" % (threads, len(filtered_files)))
        i = 0
        while i < len(filtered_files):
            ps = []
            j = 0
            while j < threads and (i + j) < len(filtered_files):
                if debug:
                    print("[%s] Creating %s of %s for file %s" % (
                i, i + j, len(filtered_files), filtered_files[i + j]))
                p = (mp.Process(target=_transform_file, args=(filtered_files[i + j],)))
                p.start()
                ps.append(p)
                j += 1

            if debug:
                print("%s process in the list to join" % len(ps))
            j = 0
            while j < threads and (i + j) < len(filtered_files):
                if debug:
                    print("[%s] Joining %s of %s for file %s" % (
                i, j, len(filtered_files), filtered_files[i + j]))
                ps[j].join()
                j += 1

            i += j

    sys.stdout.flush()


def clean_data(files_path):

    """
    Wrapper function to cleans the data and transforms it into 4D. Used to be called from either main or as block of
     the pipeline
    :param data_path: of the files to convert
    :return: MySentences class ready to be fed to Word2Vec model
    """
    print("[BLOCK] Transforming sentences to 4-dimensional lists")
    transform(files_path)
    print("[BLOCK] Done transforming data")
    sys.stdout.flush()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help="Path of the data to be used for the word embeddings"
                                                       " and clean up.", required=True)

    args = parser.parse_args()
    data_path = args.data

    print("Cleaning data from %s" % (data_path))

    begin = time()

    clean_data(data_path)


    end = time()
    print("Total processing time: %d seconds" % (end - begin))
