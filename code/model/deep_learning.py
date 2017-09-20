#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jin
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Merge,Input
from keras.layers.merge import concatenate

from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model



data = pd.read_csv('../data/quora_duplicate_questions.tsv', sep='\t')
y = data.is_duplicate.values

tk = text.Tokenizer(num_words=200000)

max_len = 40
tk.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)))
x1 = tk.texts_to_sequences(data.question1.values)
x1 = sequence.pad_sequences(x1, maxlen=max_len)

x2 = tk.texts_to_sequences(data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

word_index = tk.word_index

ytrain_enc = np_utils.to_categorical(y)

embeddings_index = {}
f = open('/path/to/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

max_features = 200000
filter_length = 5
nb_filter = 64
pool_length = 4

train_x1, holdout_x1, train_x2, holdout_x2, train_y, holdout_y = \
train_test_split(x1,x2,y,test_size = 0.98, shuffle=True, random_state=39)




model = Sequential()
print('Build model...')



embedding_layer = Embedding(len(word_index) + 1,
                            300,
                            weights=[embedding_matrix],
                            input_length=max_len,
                            trainable=False)
lstm_layer = LSTM(300, dropout=0.2, recurrent_dropout=0.2)

sequence_1_input = Input(shape=(max_len,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
result1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(max_len,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
result2 = lstm_layer(embedded_sequences_2)

merged = concatenate([result1, result2])
merged = Dropout(0.2)(merged)
merged = BatchNormalization()(merged)

merged = Dense(300, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[sequence_1_input, sequence_2_input], \
        outputs=preds)


SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

model.compile(loss='binary_crossentropy', \
    optimizer='adam', metrics=['accuracy'])


STAMP = 'lstm_%d_%d_%.2f_%.2f'%(300, 300,0.2, \
        0.2)


early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([train_x1, train_x2], y= train_y, batch_size=384, epochs=10,
                 verbose=1, validation_split=0.1, shuffle=True,\
                 callbacks=[early_stopping, model_checkpoint])


model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])