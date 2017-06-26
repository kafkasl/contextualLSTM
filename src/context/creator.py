from utils.vector_manager import VectorManager
from utils.flatten import flatten
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LsiModel, LdaMulticore, LdaModel, HdpModel
from time import time

import numpy as np
import argparse
import pickle
import sys



class TopicCreator(object):

    def __init__(self, dictionary_path, word2id, embeddings, lda=None, lsi=None):
        self.dictionary = self.load_dict(dictionary_path)
        self.word2id = VectorManager.read_vector(word2id)
        # self.word2id = self.word2id_to_id2word(word2id)
        self.embeddings = embeddings
        self.lda = lda
        self.lsi = lsi

    def load_dict(self, dict_path):
        print("[BLOCK] Loading  dictionary files from %s" % (dict_path))
        sys.stdout.flush()
        return Dictionary.load_from_text(dict_path)

    def word2id_to_id2word(self, word2id_path):

        word2id = pickle.load(open(word2id_path))
        id2word_c = [0] * len(word2id)
        for w in word2id:
            id2word_c[word2id[w]] = w
        return id2word_c


    def get_lsa_topic_embeding(self, document):
        """
        Construct a context vector by doing the weighted sum of the embeddings of the words of most relevant lsi topic
        :param document: sequence of text to get the context for
        :return: numpy array with the context
        """
        if not self.lsi:
            print("LSI model not provided")
            raise Exception("LSI model not available")

        document = [self.embeddings[int(elem)][1] for elem in document]
        corpus = [self.dictionary.doc2bow(document)]
        corpus_topics = self.lsi[corpus][0]

        values = [abs(val) for _, val in corpus_topics]
        index = values.index(max(values))

        topics = self.lsi.show_topic(index)


        embedding = np.zeros_like(self.embeddings[0][2], dtype=np.float32)
        for word, weight in topics:
            embedding = np.multiply(weight, self.embeddings[self.word2id[word]][2])

        return embedding


    def average_embeddings(self, document):
        """
        Construct a context vector by doing the average of the embeddings seen so far
        :return: numpy array with the context
        """
        if not self.lsi:
            print("LSI model not provided")
            raise Exception("LSI model not available")

        document_embeddings = [self.embeddings[int(elem)][2] for elem in document]

        embedding = np.mean(document_embeddings)

        return embedding

    def get_lda_best_topic_words(self, document):
        """
        Construct a context vector by returning the embedding of the most relevant word of the topic
        :param document: sequence of text to get the context for
        :return: numpy array with the context or unknown embedding if topic is not found
        """
        if not self.lda:
            print("LDA model not provided")
            raise Exception("LDA model not available")

        document = [self.embeddings[int(elem)][1] for elem in document]
        corpus = [self.dictionary.doc2bow(document)]
        top_topics = self.lda.top_topics(corpus, num_words=100)[0][0]

        if not top_topics[0][0] > 0.1:
            topic_word = '<unk>'
        else:
            topic_word = top_topics[0][1]

        try:
            embedding = self.embeddings[self.word2id[topic_word]][2]
        except KeyError as e:
            embedding = self.embeddings[self.word2id['<unk>']][2]
            print("Word %s not found in word2id dict, returning UNK topic (%s)" % (topic_word, e))

        return embedding

    def get_lda_topic_embedding(self, document):
        """
        Construct a context vector by doing the weighted sum of the embeddings of the 10 most relevant words of the topic
        :param document: sequence of text to get the context for
        :return: numpy array with the context
        """
        if not self.lda:
            print("LDA model not provided")
            raise Exception("LDA model not available")

        document = [self.embeddings[int(elem)][1] for elem in document]
        corpus = [self.dictionary.doc2bow(document)]
        topics = self.lda.top_topics(corpus, num_words=100)[0][0]
        top_topic = topics[0]

        if not top_topic[0] > 0:
            topic_embedding = self.embeddings[self.word2id['<unk>']][2]
        else:
            topic_embedding = np.zeros_like(self.embeddings[self.word2id['<unk>']][2], dtype=np.float32)
            for i in range(10):
                weight = topics[i][0]
                embed = self.embeddings[self.word2id[topics[i][1]]][2]
                update = np.multiply(weight, embed)
                topic_embedding = np.add(topic_embedding, update)

        return topic_embedding