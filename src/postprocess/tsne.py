from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import gzip
import pickle

# Restore the object
with gzip.open('idWordVec.pklz', 'rb') as f:
    id_word_vec = pickle.load(f)

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform([v for _, _, v in id_word_vec[0:2000]])
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

for i, word in enumerate([word for _, word, _ in id_word_vec[0:2000]]):
    plt.annotate(word, (X_tsne[i, 0], X_tsne[i, 1]))

plt.show()