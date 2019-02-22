

from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Bidirectional, Dropout, merge
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence

import keras.backend as K
import numpy as np
np.random.seed(1234)  # for reproducibility
import pickle as cPickle
import theano.tensor as T
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt


"""
Build Your Chatbot!!!
センテンスレベルの文脈ベクトル
単語レベルの意味ベクトル
"""

# params
WORD2VEC_DIMS = 100
DOC2VEC_DIMS = 300

DICTIONARY_SIZE = 10000
MAX_INPUT_LENGTH = 30
MAX_OUTPUT_LENGTH = 30

NUM_HIDDEN_UNITS = 256
BATCH_SIZE = 64
NUM_EPOCHS = 100

NUM_SUBSETS = 1

PATIENCE = 0
DROPOUT = .25
N_TEST = 100

CALL_BACKS = EarlyStopping(monitor='val_loss', patience=PATIENCE)

# files
vocabulary_file = 'vocabulary_movie'
questions_file = 'Padded_context'
answers_file = 'Padded_answers'
weights_file = 'my_model_weights20.h5'
GLOVE_DIR = './glove.6B/'

# padding and buckets

BOS = "<BOS>"
EOS = "<EOS>"
PAD = "<PAD>"

BUCKETS = [(5,10),(10,15),(15,25),(20,30)]

def print_result(input):

    ans_partial = np.zeros((1,maxlen_input))
    ans_partial[0, -1] = 2  #  the index of the symbol BOS (begin of sentence)
    for k in range(maxlen_input - 1):
        ye = model.predict([input, ans_partial])
        mp = np.argmax(ye)
        ans_partial[0, 0:-1] = ans_partial[0, 1:]
        ans_partial[0, -1] = mp
    text = ''
    for k in ans_partial[0]:
        k = k.astype(int)
        if k < (dictionary_size-2):
            w = vocabulary[k]
            text = text + w[0] + ' '
    return(text)


# ======================================================================
# Reading a pre-trained word embedding and addapting to our vocabulary:
# ======================================================================

# 辞書づくり
word2vec_index = {}
f = open(os.path.join(GLOVE_DIR, "glove.6B.100d.txt"))
for line in f:
    words = line.split()
    word = words[0]
    index = np.asarray(words[1:], dtype="float32")
    word2vec_index[word] = index
f.close()

print("The number of word vecters are: ", len(word2vec_index))

word_embedding_matrix = np.zeros((DICTIONARY_SIZE, WORD2VEC_DIMS))
# Load vocabulary
vocabulary = cPickle.load(open(vocabulary_file, 'rb'))

i = 0
for word in vocabulary:
    word2vec = word2vec_index.get(word[0])
    if word2vec is not None:
        word_embedding_matrix[i] = word2vec
    i += 1


# ======================================================================
# Keras model of the chatbot:
# ======================================================================

ADAM = Adam(lr=0.00005)

"""
Input Layer #Document*2
"""
input_context = Input(shape=(MAX_INPUT_LENGTH,), dtype="int32", name="input_context")
input_answer = Input(shape=(MAX_INPUT_LENGTH,), dtype="int32", name="input_answer")

"""
Embedding Layer: 正の整数（インデックス）を固定次元の密ベクトルに変換します．
・input_dim: 正の整数．語彙数．入力データの最大インデックス + 1．
・output_dim: 0以上の整数．密なembeddingsの次元数．
・input_length: 入力の系列長（定数）． この引数はこのレイヤーの後にFlattenからDenseレイヤーへ接続する際に必要です (これがないと，denseの出力のshapeを計算できません)．
"""
# weightが存在したら引用する
if os.path.isfile(weights_file):
    Shared_Embedding = Embedding(input_dim=DICTIONARY_SIZE, output_dim=WORD2VEC_DIMS, input_length=MAX_INPUT_LENGTH,)
else:
    Shared_Embedding = Embedding(input_dim=DICTIONARY_SIZE, output_dim=WORD2VEC_DIMS, input_length=MAX_INPUT_LENGTH,
                                 weights=[word_embedding_matrix])

"""
Shared Embedding Layer #Doc2Vec(Document*2)
"""
shared_embedding_context = Shared_Embedding(input_context)
shared_embedding_answer = Shared_Embedding(input_answer)

"""
LSTM Layer #
"""
Encoder_LSTM = LSTM(units=DOC2VEC_DIMS, init= "lecun_uniform")
Decoder_LSTM = LSTM(units=DOC2VEC_DIMS, init= "lecun_uniform")
embedding_context = Encoder_LSTM(shared_embedding_context)
embedding_answer = Decoder_LSTM(shared_embedding_answer)

"""
Merge Layer #
"""
merge_layer = merge([embedding_context, embedding_answer], mode='concat', concat_axis=1)

"""
Dense Layer #
"""
dence_layer = Dense(DICTIONARY_SIZE/2, activation="relu")(merge_layer)

"""
Output Layer #
"""
outputs = Dense(DICTIONARY_SIZE, activation="softmax")(dence_layer)

"""
Modeling
"""
model = Model(input=[input_context, input_answer], output=[outputs])
model.compile(loss="categorical_crossentropy", optimizer=ADAM)

if os.path.isfile(weights_file):
    model.load_weights(weights_file)


# ======================================================================
# Loading the data:
# ======================================================================

Q = cPickle.load(open(questions_file, 'rb'))
A = cPickle.load(open(answers_file, 'rb'))
N_SAMPLES, N_WORDS = A.shape

Q_test = Q[0:N_TEST,:]
A_test = A[0:N_TEST,:]
Q = Q[N_TEST + 1:,:]
A = A[N_TEST + 1:,:]

print("Number of Samples = %d"%(N_SAMPLES - N_TEST))
Step = np.around((N_SAMPLES - N_TEST) / NUM_SUBSETS)
SAMPLE_ROUNDS = Step * NUM_SUBSETS


# ======================================================================
# Bot training:
# ======================================================================

x = range(0, NUM_EPOCHS)
VALID_LOSS = np.zeros(NUM_EPOCHS)
TRAIN_LOSS = np.zeros(NUM_EPOCHS)

for n_epoch in range(NUM_EPOCHS):
    # Loop over training batches due to memory constraints
    for n_batch in range(0, SAMPLE_ROUNDS, Step):

        Q2 = Q[n_batch:n_batch+Step]
        s = Q2.shape
        counter = 0
        for id, sentence in enumerate(A[n_batch:n_batch+Step]):
            l = np.where(sentence==3)  #  the position od the symbol EOS
            limit = l[0][0]
            counter += limit + 1

        question = np.zeros((counter, MAX_INPUT_LENGTH))
        answer = np.zeros((counter, MAX_INPUT_LENGTH))
        target = np.zeros((counter, DICTIONARY_SIZE))

        # Loop over the training examples:
        counter = 0
        for i, sentence in enumerate(A[n_batch:n_batch+Step]):
            ans_partial = np.zeros((1, MAX_INPUT_LENGTH))

            # Loop over the positions of the current target output (the current output sequence)
            l = np.where(sent==3)  #  the position of the symbol EOS
            limit = l[0][0]

            for k in range(1, limit+1):
                # Mapping the target output (the next output word) for one-hot codding:
                target = np.zeros((1, DICTIONARY_SIZE))
                target[0, sentence[k]] = 1

                # preparing the partial answer to input:
                ans_partial[0,-k:] = sentence[0:k]

                # training the model for one epoch using teacher forcing:

                question[counter, :] = Q2[i:i+1]
                answer[counter, :] = ans_partial
                target[counter, :] = target
                counter += 1

        print('Training epoch: %d, Training examples: %d - %d'%(n_epoch, n_batch, n_batch + Step))
        model.fit([question, answer], target, batch_size=BATCH_SIZE, epochs=1)

        test_input = Q_test[41:42]
        print(print_result(test_input))
        train_input = Q_test[41:42]
        print(print_result(train_input))

    model.save_weights(weights_file, overwrite=True)
