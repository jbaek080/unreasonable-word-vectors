from __future__ import absolute_import
import gensim

import tensorflow as tf
from keras import backend as K
from keras.losses import binary_crossentropy
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense, merge, dot
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.objectives import mse
from scipy.spatial.distance import cosine

from sentences_generator import Sentences, GutenbergSentences, BrownSentences, ReutersSentences
import vocab_generator as V_gen
import save_embeddings as S
import global_settings as G
import sys
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

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
    #sentences = Sentences("WestburyLab.Wikipedia.Corpus.txt")
    sentences = GutenbergSentences()
    vocabulary = dict()
    V_gen.build_vocabulary(vocabulary, sentences)
    V_gen.filter_vocabulary_based_on(vocabulary, G.min_count)
    reverse_vocabulary, non_reverse_vocabulary = V_gen.generate_vocabulary_lookups(vocabulary, "vocab.txt")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    def load_google_news_model():
        print("Loading word2vec google news model")
        loc = './GoogleNews-vectors-negative300.bin'
        w2vmodel = gensim.models.KeyedVectors.load_word2vec_format(loc, binary=True)
        print("loading complete")
        g_embed_dim = 300
        unk = [0 for _ in range(300)]
        embedding = [unk, unk]
        for i in range(2, len(non_reverse_vocabulary) + 2):
            word = non_reverse_vocabulary[i]
            if word in w2vmodel:
                embedding.append(w2vmodel[word])
            else:
                r = np.random.uniform(-1.0/2.0/g_embed_dim, 1.0/2.0/g_embed_dim, (g_embed_dim,))
                embedding.append(r)
        embedding.append(np.random.uniform(-1.0/2.0/g_embed_dim, 1.0/2.0/g_embed_dim, (g_embed_dim,)))
        embedding.append(np.random.uniform(-1.0/2.0/g_embed_dim, 1.0/2.0/g_embed_dim, (g_embed_dim,)))
        embedding = np.array(embedding, dtype=np.float64)
        print("Loading complete")
        return embedding

    if sys.argv[1] == "vanilla":
        embedding = load_google_news_model()

    elif sys.argv[1] == "new":
        embedding = load_google_news_model()
        #embedding = np.random.uniform(-1.0/2.0/G.embedding_dimension, 1.0/2.0/G.embedding_dimension, (G.vocab_size+3, G.embedding_dimension)) 
        k = G.window_size # context windows size
        context_size = 2*k

        def model():
            word_input = Input(shape=(1,))
            context_input = Input(shape=(context_size,))

            def regularize(weight_matrix):
                return 0.1 * (weight_matrix[reverse_vocabulary["man"]] - weight_matrix[reverse_vocabulary["woman"]]) ** 2

            shared_embedding_layer = Embedding(input_dim=(G.vocab_size+3), output_dim=G.embedding_dimension, embeddings_regularizer = regularize, weights=[embedding])
            word_embedding = shared_embedding_layer(word_input)
            context_embeddings = shared_embedding_layer(context_input)
            cbow = Lambda(lambda x: K.mean(x, axis=1), output_shape=(G.embedding_dimension,))(context_embeddings)
            word_context_product = dot([word_embedding, cbow], axes=-1)
            #loss_out = Lambda(customized_loss, output_shape=(1,), name='joint_loss')([word_input, word_context_product, shared_embedding_layer.get_weights()[0]])
            model = Model(inputs=[word_input, context_input], outputs=[word_context_product])

            return model, word_embedding, shared_embedding_layer

        def custom_loss_wrapper(shared_embedding_layer, alpha=0.5):
            def custom_loss(y_true, y_pred):
                #return dot([word_embedding[reverse_vocabulary['woman']],
                #        word_embedding[reverse_vocabulary['man']]], axes=-1)
                print(shared_embedding_layer.get_weights()[0][reverse_vocabulary['woman']])
                return binary_crossentropy(y_true, y_pred) * 0 - alpha * similarity('woman', 'man', shared_embedding_layer.get_weights()[0])
            return custom_loss

        def customized_loss(args):
            y_true, y_pred, word_embedding = args
            alpha = 0.1
            #return binary_crossentropy(y_true, y_pred) - alpha * cos_similarity(woman, man)
            return dot([word_embedding[reverse_vocabulary['woman']],
                        word_embedding[reverse_vocabulary['man']]], axes=-1)
            #return binary_crossentropy(y_true, y_pred) * 0 - alpha * similarity('woman', 'man', word_embedding)

        m, w, s = model()
        #m.compile(optimizer='rmsprop', loss={'joint_loss': lambda y_true, y_pred: y_pred})
        m.compile(optimizer='rmsprop', loss=binary_crossentropy)
        print(m.summary())

        gen = V_gen.pretraining_batch_generator(sentences, vocabulary, reverse_vocabulary) 
        m.fit_generator(gen, steps_per_epoch=len(vocabulary)/5, epochs=1)
        # Save the trained embedding
        pickle.dump(shared_embedding_layer.get_weights()[0], open("embeddings.pkl", "wb"))
        S.save_embeddings("embedding.txt", shared_embedding_layer.get_weights()[0], vocabulary)

    else:
        print("loading embeddings", sys.argv[1])
        word_embeddings = pickle.load(open(sys.argv[1], 'rb'))
        print("embeddings loaded")
        if (len(sys.argv) == 2):
            tsne = TSNE(perplexity=2,verbose=2,random_state=0,n_iter=3000).fit(word_embeddings[:1000,:])
            xdata = tsne.embedding_[:, 0]
            ydata = tsne.embedding_[:, 1]
            plt.scatter(xdata, ydata)
            for label, x, y in zip(vocabulary, xdata, ydata): 
                plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
            plt.show()



