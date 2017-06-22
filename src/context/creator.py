from utils.vector_manager import VectorManager
from utils.flatten import flatten
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


def get_lsa_topic_embeding(document, dictionary, lsi, word2id, embeddings):
    """
    Construct a context vector by doing the weighted sum of the embeddings of the words of most relevant lsi topic
    :param document: sequence of text to get the context for
    :param dictionary: dictionary to turn the document into a bag of words
    :param lsi: lsi model used for inferring the topics
    :param word2id: word2id mappings (mapping to precomputed embeddings)
    :param embeddings: word embeddings
    :return: numpy array with the context
    """
    corpus = [dictionary.doc2bow(document.split())]
    corpus_topics = lsi[corpus][0]

    values = [abs(val) for _, val in corpus_topics]
    index = values.index(max(values))

    topics = lsi.show_topic(index)


    embedding = np.zeros_like(embeddings[0][2], dtype=np.float32)
    for word, weight in topics:
        embedding = np.multiply(weight, embeddings[word2id[word]][2])

    return embedding

def get_lda_best_topic_words(document, dictionary, lda, word2id, embeddings):
    """
    Construct a context vector by returning the embedding of the most relevant word of the topic
    :param document: sequence of text to get the context for
    :param dictionary: dictionary to turn the document into a bag of words
    :param lda: LDA model used for inferring the topics
    :param word2id: word2id mappings (mapping to precomputed embeddings)
    :param embeddings: word embeddings
    :return: numpy array with the context or unknown embedding if topic is not found
    """
    corpus = [dictionary.doc2bow(document.split())]
    top_topics = lda.top_topics(corpus, num_words=100)[0][0]

    if not top_topics[0][0] > 0:
        topic_word = '<unk>'
    else:
        topic_word = [w for _, w in top_topics[0:n]]

    try:
        embedding = embeddings[word2id[topic_word]][2]
    except KeyError as e:
        embedding = embeddings[word2id['<unk>']][2]
        print("Word %s not found in word2id dict, returning UNK topic (%s)" % (topic_word, e))

    return embedding

def get_lda_topic_embedding(document, dictionary, lda, word2id, embeddings):
    """
    Construct a context vector by doing the weighted sum of the embeddings of the 10 most relevant words of the topic
    :param document: sequence of text to get the context for
    :param dictionary: dictionary to turn the document into a bag of words
    :param lda: LDA model used for inferring the topics
    :param word2id: word2id mappings (mapping to precomputed embeddings)
    :param embeddings: word embeddings
    :return: numpy array with the context
    """
    corpus = [dictionary.doc2bow(document.split())]
    topics = lda.top_topics(corpus, num_words=100)[0][0]
    top_topic = topics[0]

    if not top_topic[0] > 0:
        topic_embedding  = embeddings[word2id['<unk>']][2]
    else:
        topic_embedding  = np.zeros_like(embeddings[word2id['<unk>']][2], dtype=np.float32)
        for i in range(10):
            weight = topics[i][0]
            embed = embeddings[word2id[topics[i][1]]][2]
            update = np.multiply(weight, embed)
            topic_embedding = np.add(topic_embedding, update)

    return topic_embedding


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-m', '--models', type=str, help="Directory where the models are be stored.", required=True)
#     parser.add_argument('-e', '--embeddings', type=str, help="Embeddings path (id_word_vec.pklz)", required=True)
#     parser.add_argument('-w', '--word2id_path', type=str, help="Word2ID vector to be used for doc translation.",
#                         required=True, default=None)
#     parser.add_argument('-i', '--id_word', type=str, help="Id2Word vector path ['wiki_en_wordids.txt'].",
#                         required=True, default=None)
#
#     args = parser.parse_args()
#
#     model_path = args.models
#     id2word_path = args.id_word
#     word2id_path = args.word2id_path
#
#     begin = time()
#
#     dictionary = load_dict(id2word_path)
#     id2word = word2id_to_id2word(word2id_path)
#     lda = LdaModel.load(model_path)
#
#
#     corpus = [dictionary.doc2bow(doc.split()) for doc in [demo]]
#
#
#     print lda[corpus]
#
#
#     end = time()
#     print("Total processing time: %d seconds" % (end - begin))
