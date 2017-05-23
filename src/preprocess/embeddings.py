import gensim
import multiprocessing
import os
import re
import argparse
from utils.vector_manager import VectorManager


from pattern.en import tokenize
from time import time


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext


def remove_title(text):
   index = text.find("\n\n")
   if index != -1 :
         return text[index+2:]
   else :
         return text


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def transform(self):
        for root, dirs, files in os.walk(self.dirname):
            filtered_files = [file for file in files if is_number(file.split("_")[1]) and len(file.split("_")) == 2]
            for filename in filtered_files:
                file_path = root + '/' + filename
                data = open(file_path).read()
                file_out = "%s_clean" % file_path
                docs = data.split("</doc>")
                file_list = []
                for doc in [d.strip() for d in docs if d.strip()]:
                    doc_list = []
                    paragraphs = [tokenize(par) for par in remove_title(cleanhtml(doc)).strip().split("\n\n")]

                    for p in paragraphs:
                        par_list = []
                        # print("Paragraph: %s" % p)
                        for sent in p:
                            line = [word for word in sent.lower().split()
                                    if word.isalpha() or is_number(word)]
                            if line:
                                par_list.append(line)
                        doc_list.append(par_list)
                    file_list.append(doc_list)
                VectorManager.write_file(file_out, file_list)

    def __iter__(self):
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

    sentences = MySentences(data_path)
    print("Transforming data 4-dimensional lists")
    sentences.transform()
    print("Done transforming data, proceeding to create embeddings")

    return sentences


def create_embeddings(sentences, emb_size, min_count):

    model = gensim.models.Word2Vec(sentences,
                                   size=emb_size,
                                   window=10,
                                   min_count=min_count,
                                   workers=multiprocessing.cpu_count())
    print("Saving embeddings model...")
    model.save("models/word2vec_gensim_%s" % emb_size)
    model.wv.save_word2vec_format("models/word2vec_org_%s" % emb_size,
                                  "models/vocabulary_%s" % emb_size,
                                  binary=False)
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help="Path of the data to be used for the word embeddings"
                                                       " and clean up.", required=True)
    parser.add_argument('-s', '--size', type=int, help="Size of the word embeddings.", default=200, required=False)

    args = parser.parse_args()
    data_path = args.data
    emb_size = args.size

    print("Parsing data from %s\nEmbedding size: %s" % (data_path, emb_size))

    begin = time()

    sentences = clean_data(data_path)

    min_count = 200

    create_embeddings(sentences, emb_size, min_count)

    end = time()
    print("Total processing time: %d seconds" % (end - begin))
