# -*- coding: utf-8 -*-


from gensim.models.keyedvectors import KeyedVectors
from utils.vector_manager import VectorManager
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse


def plot_tsne(id_word_vec):

    """
    Compute the t-SNE dimensionality reduction values of input parameter and plot them in 2D
    :param id_word_vec: vector containing the tuples (id, word, embedding) to be plotted
    """
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform([v for _, _, v in id_word_vec])
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

    for i, word in enumerate([word for _, word, _ in id_word_vec]):
        plt.annotate(word, (X_tsne[i, 0], X_tsne[i, 1]))

    plt.show()


def subset(initial_word, id_word_vec, wv, max):
    """
    Get a subset of max number of words using cosmul distance starting from initial_word
    :param initial_word: first word to be used to find nearest ones
    :param id_word_vec: vector containing the tuples (id, word, embedding) for each word
    :param wv: gensim word embeddings model
    :param max: number of words to return
    :return: list of tuples (id, word, embedding)
    """
    words = [initial_word]
    subset = []
    while len(words) > 0 and len(subset) < max:
        w = words.pop()
        sim = wv.similar_by_word(w)
        ws = [w for w, _ in sim]
        similars = [s for s in ws if s not in subset]
        subset.extend(similars)
        words.extend(similars)

    final_set = [(i, w, v) for i, w, v in id_word_vec if w in subset]
    return final_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--id_word_vec', type=str, help="Path of id <-> word <-> embedding vector", required=True)
    parser.add_argument('-w', '--word_vectors', type=str, help="Path of LM to perform the tests upon", required=True)

    args = parser.parse_args()

    # Arguments parsing
    wv_path = args.word_vectors
    path = args.id_word_vec

    print("Loading model...")
    wv = KeyedVectors.load_word2vec_format(wv_path, binary=False)

    print("Loading id-word-vec...")
    id_word_vec = VectorManager.read_vector(path)

    print("Finding subset to plot")
    initial_word = 'jupiter'
    max_elements = 500
    sb = subset(initial_word, id_word_vec, wv, max_elements)

    print("Plotting subset of words...")
    # Plot t-SNE
    plot_tsne(sb)


