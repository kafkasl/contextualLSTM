from utils.vector_manager import VectorManager
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LsiModel, LdaMulticore, LdaModel, HdpModel
from time import time

import numpy as np
import argparse
import pickle
import sys


def load_dict(id2word_path):
    print("[BLOCK] Loading  dictionary files from %s" % (id2word_path))
    sys.stdout.flush()
    dictionary = Dictionary.load_from_text(id2word_path)


    return dictionary


def word2id_to_id2word(word2id_path):

    word2id = pickle.load(open(word2id_path))
    id2word_c = [0] * len(word2id)
    for w in word2id:
        id2word_c[word2id[w]] = w
    return id2word_c



def print_lsa_topic(document, dictionary, lsi):
    corpus = [dictionary.doc2bow(document.split())]
    topics = lsi[corpus]
    topics = topics[0]  # Only one document

    values = [abs(val) for _, val in topics]
    index = values.index(max(values))
    # print(values)
    print(topics[index], lsi.print_topic(index))


def print_hdp(document, dictionary, hdp):
    corpus = [dictionary.doc2bow(document.split())]
    corpus_hdp = hdp[corpus]

    for doc in corpus_hdp:
        values = [abs(val) for _, val in doc]
        index = values.index(max(values))
        # print(values)
        print(doc[index], hdp.print_topic(index))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help="Directory where the model is stored.", required=True)
    parser.add_argument('-e', '--embeddings', type=str, help="Embeddings path (id_word_vec.pklz)", required=True)
    parser.add_argument('-w', '--word2id_path', type=str, help="Word2ID vector to be used for doc translation.",
                        required=True, default=None)
    parser.add_argument('-i', '--id_word', type=str, help="Id2Word vector path ['wiki_en_wordids.txt'].",
                        required=True, default=None)

    args = parser.parse_args()

    model_path = args.model
    id2word_path = args.id_word
    word2id_path = args.word2id_path
    emb_path = args.embeddings

    begin = time()

    dictionary = load_dict(id2word_path)
    id2word = word2id_to_id2word(word2id_path)
    w2Id = VectorManager.read_vector(word2id_path)
    embeddings = VectorManager.read_vector(emb_path)

    demo1 = "the roman consul is normally a notable person from the senate elected " \
            "by direct voting of the italic tribes"

    data = open("../data/small/AA/wiki_01_clean_simple").read().split("<eop>")
    s1 = data[0].split("<eos>")[0]
    data = open("../data/small/AA/wiki_00_clean_simple").read().split("<eop>")
    s2 = data[0].split("<eos>")[0]
    data = open("../data/small/AB/wiki_00_clean_simple").read().split("<eop>")
    s3 = data[0].split("<eos>")[0]
    data = open("../data/small/AB/wiki_01_clean_simple").read().split("<eop>")
    s4 = data[0].split("<eos>")[0]


    if "lda" in model_path:
        lda = LdaModel.load(model_path)
        print("Demo 1:\n%s" % demo1)
        print(get_lda_best_topic_words(demo1, dictionary, lda))
        print("Demo 2:\n%s" % s1)
        print(get_lda_best_topic_words(s1, dictionary, lda))
        print("Demo 3:\n%s" % s2)
        print(get_lda_best_topic_words(s2, dictionary, lda))
        print("Demo 4:\n%s" % s3)
        print(get_lda_best_topic_words(s3, dictionary, lda))
        print("Demo 5:\n%s" % s4)
        print(get_lda_best_topic_words(s4, dictionary, lda))
    elif "lsa" in model_path:
        lsi = LsiModel.load(model_path)
        print("Demo 1:\n%s" % demo1)
        print(print_lsa_topic(demo1, dictionary, lsi))
        print("Demo 2:\n%s" % s1)
        print(print_lsa_topic(s1, dictionary, lsi))
        print("Demo 3:\n%s" % s2)
        print(print_lsa_topic(s2, dictionary, lsi))
        print("Demo 4:\n%s" % s3)
        print(print_lsa_topic(s3, dictionary, lsi))
        print("Demo 5:\n%s" % s4)
        print(print_lsa_topic(s4, dictionary, lsi))
        print(get_lsa_topic_embeding(s4, dictionary, lsi, w2Id, embeddings))
    elif "hdp" in model_path:
        hdp = HdpModel.load(model_path)
        print("Demo 1:\n%s" % demo1)
        print(print_hdp(demo1, dictionary, hdp))


    end = time()
    print("Total processing time: %d seconds" % (end - begin))
