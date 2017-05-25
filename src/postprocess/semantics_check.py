from gensim.models.keyedvectors import KeyedVectors

import gensim
import os
import argparse

module_path = "%s/test" % os.path.dirname(gensim.__file__)


def semantics_checks():
    print("Operations using multiplicative combination objective:")
    w = wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
    print(" * King + Woman - Man = %s" % w[0])
    w = wv.most_similar_cosmul(positive=['baghdad', 'england'], negative=['london'])
    print(" * Baghdad + England - London = %s" % w[0])


wv.evaluate_word_pairs(os.path.join(module_path, 'test_data','wordsim353.tsv'))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--word_vectors', type=str, help="Path of LM to perform the tests upon", required=True)

    args = parser.parse_args()

    # Arguments parsing
    wv_path = args.word_vectors
    wv = KeyedVectors.load_word2vec_format(wv_path, binary=False)


