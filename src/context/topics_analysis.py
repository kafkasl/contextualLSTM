from utils.vector_manager import VectorManager
from utils.flatten import flatten
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LsiModel, LdaMulticore, LdaModel, HdpModel
from time import time

import multiprocessing as mp

import argparse
import os
import sys
import bz2

stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
              "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between",
              "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
              "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
              "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
              "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
              "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me",
              "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
              "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
              "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
              "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
              "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
              "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
              "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
              "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
              "your", "yours", "yourself", "yourselves"]


def get_file_as_list(filename):
    words_list = VectorManager.parse_into_list(VectorManager.read_vector(filename))
    words_list = [w for w in words_list if w not in stop_words]
    return words_list


def get_lists(data_path):
    filepaths = []
    for root, dirs, files in os.walk(data_path):
        filepaths.extend(["%s/%s" % (root, file) for file in files if file.endswith("_clean")])

    p = mp.Pool(mp.cpu_count() * 2)
    files_list = p.map(get_file_as_list, filepaths)

    return filepaths, files_list


def get_corpus_and_dict(data_path):
    print("[BLOCK] Getting corpus and dictionary files from %s" % (data_path))
    sys.stdout.flush()

    file_paths, files_list = get_lists(data_path)

    print("[BLOCK] Building dictionary with %s documents" % len(files_list))
    sys.stdout.flush()

    dictionary = Dictionary(files_list)

    print("[BLOCK] Filtering out %s (0.1)" % (int(len(dictionary)*0.1)))
    sys.stdout.flush()

    dictionary.filter_n_most_frequent(int(len(dictionary)*0.1))

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(doc) for doc in files_list]

    return corpus, dictionary


def load_corpus_and_dict(corpus_path, id2word_path):
    print("[BLOCK] Loading  corpus and dictionary files from %s and %s" % (data_path, id2word_path))
    sys.stdout.flush()
    dictionary = Dictionary.load_from_text(id2word_path)

    print("[BLOCK] Loading corpus iterator")
    sys.stdout.flush()
    #mm = gensim.corpora.MmCorpus(corpus_path)
    corpus = MmCorpus(bz2.BZ2File(corpus_path)) # use this if you compressed the TFIDF output (recommended)

    return corpus, dictionary


def topic_analysis(corpus, dictionary, models_path, technique):

    import uuid
    uuid = str(uuid.uuid4())
    print("[BLOCK] Starting models for context")
    sys.stdout.flush()

    if technique == "all" or technique == "hdp":
        t1 = time()
        # HDP model
        model = HdpModel(corpus, id2word=dictionary)
        model.save("%s/hdp_%s" % (models_path, uuid))
        del model
        t2 = time()
        print("[BLOCK] Training time for HDP model: %s" % (round(t2-t1, 2)))
        sys.stdout.flush()

    if technique == "all" or technique == "ldap":
        t1 = time()
        # Parallel LDA model
        model = LdaMulticore(corpus, id2word=dictionary, num_topics=100,  workers=23, passes=20)
        model.save("%s/lda_parallel_%s" % (models_path, uuid))
        del model
        t2 = time()
        print("[BLOCK] Training time for LDA multicore: %s" % (round(t2-t1, 2)))
    sys.stdout.flush()

    if technique == "all" or technique == "lsa":
        t1 = time()
        # LSA model
        model = LsiModel(corpus, id2word=dictionary, num_topics=400)
        model.save("%s/lsa_%s" % (models_path, uuid))
        del model
        t2 = time()
        print("[BLOCK] Training time for LSA: %s" % (round(t2-t1, 2)))
        sys.stdout.flush()

    if technique == "all" or technique == "ldao":
        t1 = time()
        # Online LDA model
        model = LdaModel(corpus, id2word=dictionary, num_topics=100, update_every=1, chunksize=10000, passes=5)
        model.save("%s/lda_online_%s" % (models_path, uuid))
        t2 = time()
        print("[BLOCK] Training time for LDA online: %s" % (round(t2-t1, 2)))
        sys.stdout.flush()

    if technique == "all" or technique == "lda":
        t1 = time()
        # Offline LDA model
        model = LdaModel(corpus, id2word=dictionary, num_topics=100,  update_every=0, passes=20)
        model.save("%s/lda_offline_%s" % (models_path, uuid))
        del model
        t2 = time()
        print("[BLOCK] Training time for LDA offline: %s" % (round(t2-t1, 2)))
        sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help="Path of the data to be translated with word2id vector."
                                                       " and clean up.", required=True)
    parser.add_argument('-m', '--models', type=str, help="Directory were the models will be stored.", required=True)
    parser.add_argument('-w', '--word_vector', type=str, help="Word2ID vector to be used for doc translation.",
                        required=False, default=None)
    parser.add_argument('-c', '--corpus_path', type=str, help="Corpus iterator path [wiki_en_tfidf.mm.bz2].",
                        required=False, default=None)
    parser.add_argument('-i', '--id_word', type=str, help="Id2Word vector path ['wiki_en_wordids.txt'].",
                        required=False, default=None)
    parser.add_argument('-t', '--technique', type=str, help="Technique used for topic modeling. Available options all,"
                        "hierarchical dirichlet process (hdp), latent dirichlet allocation (lda), lda multicore (ldap)"
                        "latent semantic anaylisis (lsa), lda online (ldao)", required=False, default="all")

    args = parser.parse_args()
    data_path = args.data
    models_path = args.models
    word2id_file = args.word_vector
    corpus_path = args.corpus_path
    id2word_path = args.id_word
    technique = args.technique

    begin = time()

    if word2id_file:
        w2Id = VectorManager.read_vector(word2id_file)

    if corpus_path and id2word_path:
        corpus, dictionary = load_corpus_and_dict(corpus_path, id2word_path)
    else:
        corpus, dictionary = get_corpus_and_dict(data_path)

    topic_analysis(corpus, dictionary, models_path, technique)

    end = time()
    print("Total processing time: %d seconds" % (end - begin))
