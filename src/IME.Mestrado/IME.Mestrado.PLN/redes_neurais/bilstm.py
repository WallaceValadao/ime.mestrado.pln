from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import SpatialDropout1D

import tensorflow as tf
import numpy as np

top_words = 50000
embedding_vecor_length = 32

class BiLstmClassification():

    def __init__(self, config, activation, epochs = 10, verbose=0):
        self.epochs = epochs
        self.activation = activation
        self.verbose = verbose
        self.configs = config


    def _createInstance(self, maxReviewLength):
        self.max_review_length = maxReviewLength
        self.model = Sequential()
        self.model.add(Embedding(top_words, embedding_vecor_length, input_length=self.max_review_length))
        self.model.add(SpatialDropout1D(0.5))
        self.model.add(Bidirectional(LSTM(100, dropout=0.5, recurrent_dropout=0.5)))
        self.model.add(Dense(self.configs.numero_classes, activation=self.activation))
        #run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    def fit(self, X_train, Y_train, maxReviewLength):
        self._createInstance(maxReviewLength)

        #X_train = sequence.pad_sequences(X_train, maxlen=self.max_review_length)
        Y_train = tf.keras.utils.to_categorical(Y_train, 2)

        self.model.fit(np.array(X_train), np.array(Y_train), verbose=self.verbose, epochs=self.epochs, batch_size=16,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


    def predict(self, x_teste): 
        x_teste = sequence.pad_sequences(x_teste, maxlen=self.max_review_length)

        results = self.model.predict(x_teste)

        resultados = []

        for item in results:
            resultados.append(np.argmax(item))

        return resultados
