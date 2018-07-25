from __future__ import absolute_import

from keras import backend as K
from keras.losses import binary_crossentropy
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense, merge, dot
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.objectives import mse
from scipy.spatial.distance import cosine

from sentences_generator import Sentences, GutenbergSentences, BrownSentences
import vocab_generator as V_gen
import save_embeddings as S
import global_settings as G
import sys
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

sentences = BrownSentences()
vocabulary = dict()
V_gen.build_vocabulary(vocabulary, sentences)
V_gen.filter_vocabulary_based_on(vocabulary, G.min_count)
reverse_vocabulary, non_reverse_vocabulary = V_gen.generate_vocabulary_lookups(vocabulary, "vocab.txt")

def cos_similarity(a, b):
    return np.abs(-cosine(a, b) + 1)

def similarity(w1, w2, word_embeddings):
    return cos_similarity(word_embeddings[reverse_vocabulary[w1]], word_embeddings[reverse_vocabulary[w2]])

def n_most_similar(word_inp, word_embeddings, n=5):
    word_to_similarity = {}
    for i in range(2, len(word_embeddings) - 2):
        word = non_reverse_vocabulary[i]
        word_vector = word_embeddings[i]
        sim = similarity(word_inp, word, word_embeddings)
        word_to_similarity[word] = sim

    top_n = sorted(word_to_similarity.items(), key=lambda x: -x[1])[:n]
    return top_n

def analogy(a, b, c, word_embeddings, n = 5):
    word_to_similarity = {}
    import pdb; pdb.set_trace()
    a = word_embeddings[reverse_vocabulary[a]]
    b = word_embeddings[reverse_vocabulary[b]]
    c = word_embeddings[reverse_vocabulary[c]]
    for i in range(2, len(word_embeddings) - 2):
        word = non_reverse_vocabulary[i]
        d = word_embeddings[i]
        sim = cos_similarity(a - b, c - d)
        word_to_similarity[word] = sim
    top_n = sorted(word_to_similarity.items(), key=lambda x: -x[1])[:n]
    return top_n
    


def load_embeddings(filename):
    embeddings = []
    with open(filename) as f:
        f.readline()
        for line in f:
            word_and_vec = embeddings.split("\t")
            word = word_and_vec[0]
            vec = [float(i) for i in word_and_vec[1].split(" ")]
            embeddings.append(vec)

    return embeddings

if __name__ == "__main__":
    if sys.argv[1] == "new":
        k = G.window_size # context windows size
        context_size = 2*k
        embedding = np.random.uniform(-1.0/2.0/G.embedding_dimension, 1.0/2.0/G.embedding_dimension, (G.vocab_size+3, G.embedding_dimension))

        # Creating CBOW model
        # Model has 3 inputs
        # Current word index, context words indexes and negative sampled word indexes
        word_index = Input(shape=(1,))
        context = Input(shape=(context_size,))
        negative_samples = Input(shape=(G.negative,))
        # All the inputs are processed through a common embedding layer
        shared_embedding_layer = Embedding(input_dim=(G.vocab_size+3), output_dim=G.embedding_dimension, weights=[embedding])
        word_embedding = shared_embedding_layer(word_index)
        context_embeddings = shared_embedding_layer(context)
        negative_words_embedding = shared_embedding_layer(negative_samples)
        # Now the context words are averaged to get the CBOW vector
        cbow = Lambda(lambda x: K.mean(x, axis=1), output_shape=(G.embedding_dimension,))(context_embeddings)
        # The context is multiplied (dot product) with current word and negative sampled words
        word_context_product = dot([word_embedding, cbow], axes=-1)
        negative_context_product = dot([negative_words_embedding, cbow], axes=-1)
        # The dot products are outputted
        model = Model(inputs=[word_index, context, negative_samples], outputs=[word_context_product, negative_context_product])
        i_woman = reverse_vocabulary['woman']
        i_man = reverse_vocabulary['man']

        def loss_mse(y_true, y_pred, alpha = .1):
            word_embeddings = shared_embedding_layer.get_weights()[0]
            woman = word_embeddings[i_woman]
            man = word_embeddings[i_man]
            return binary_crossentropy(y_true, y_pred)# - alpha * cos_similarity(woman, man)

        model.compile(optimizer='rmsprop', loss=loss_mse)
        print(model.summary())

        word_embeddings = shared_embedding_layer.get_weights()[0]

        gen = V_gen.pretraining_batch_generator(sentences, vocabulary, reverse_vocabulary) 
        model.fit_generator(gen, steps_per_epoch=len(vocabulary), epochs=1)
        # Save the trained embedding
        pickle.dump(shared_embedding_layer.get_weights()[0], open("embeddings.pkl", "wb"))
        S.save_embeddings("embedding.txt", shared_embedding_layer.get_weights()[0], vocabulary)

        word_embeddings = shared_embedding_layer.get_weights()[0]
        print("similarity between woman and man: ", str(cos_similarity(word_embeddings[i_woman], word_embeddings[i_man])))
        print("similarity between husband and wife: ", str(cos_similarity(word_embeddings[reverse_vocabulary['husband']], word_embeddings[reverse_vocabulary['wife']])))
    else:
        print("loading embeddings", sys.argv[1])
        word_embeddings = pickle.load(open(sys.argv[1], 'rb'))
        print("embeddings loaded")
        pca = PCA(n_components=20).fit(word_embeddings)
        tsne = TSNE().fit(pca.components_)
        xdata = tsne.embedding_[:, 0]
        ydata = tsne.embedding_[:, 1]
        plt.scatter(xdata, ydata)



