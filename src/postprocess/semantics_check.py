from gensim.models.keyedvectors import KeyedVectors
from pprint import pprint
import gensim
import os
import argparse

module_path = "%s/test" % os.path.dirname(gensim.__file__)


def semantics_checks(wv):
    """
    Perform some semantics check to see that the generated word vectors are sensible
    :param wv: word vectors of the embeddings
    """
    print("Operations using multiplicative combination objective:")
    w = wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
    print(" * King + Woman - Man = %s [%s]" % (w[0][0], w[0][1]))
    w = wv.most_similar_cosmul(positive=['baghdad', 'england'], negative=['london'])
    print(" * Baghdad + England - London = %s [%s]" % (w[0][0], w[0][1]))

    print("\n * Most similar words to Paris:")
    pprint(wv.most_similar_cosmul('paris'))

    print("\n * Most similar words to Jupiter:")
    pprint(wv.most_similar_cosmul('jupiter'))

    print("\n * Most similar words to Zeus:")
    pprint(wv.most_similar_cosmul('zeus'))


def compute_accuracies(wv):
    """
    Compute the accuracy of parameter word embeddings with 5 semantic and 9 grammatical relations
    :param wv: word vectors of the embeddings
    """
    acc = wv.accuracy(os.path.join(module_path, 'test_data', 'questions-words.txt'))
    for sec in acc:
        correct = len(sec['correct'])
        incorrect = len(sec['incorrect'])
        total = correct + incorrect
        ac = correct / float(total)
        print("\n[%s]\n\tAccuracy [%s] %s/%s" % (sec['section'].title(), round(ac, 2), correct, total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--word_vectors', type=str, help="Path of LM to perform the tests upon", required=True)

    args = parser.parse_args()

    # Arguments parsing
    wv_path = args.word_vectors

    print("Loading model...")
    wv = KeyedVectors.load_word2vec_format(wv_path, binary=False)

    # Some semantic examples
    semantics_checks(wv)

    # Compute and print questions accuracies
    compute_accuracies(wv)


