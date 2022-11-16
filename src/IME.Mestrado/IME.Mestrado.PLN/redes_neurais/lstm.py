from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import SpatialDropout1D

import tensorflow as tf
import numpy as np

import configs as ref_config

top_words = 50000
embedding_vecor_length = 16

class LstmClassification():

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
        #self.model.add(LeakyReLU())
        self.model.add(LSTM(maxReviewLength, dropout=0.5, recurrent_dropout=0.5))
        self.model.add(Dense(self.configs.numero_classes, activation=self.activation))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        
    def fit(self, X_train, Y_train, maxReviewLength):
        self._createInstance(maxReviewLength)

        X_train = sequence.pad_sequences(X_train, maxlen=self.max_review_length)
        Y_train = tf.keras.utils.to_categorical(Y_train, 2)

        self.model.fit(X_train, Y_train, verbose=self.verbose, 
                       epochs=2, batch_size=16,
                       validation_split=0.1,
                       steps_per_epoch=100, 
                       validation_steps=10,
                       callbacks=[EarlyStopping(monitor='val_loss', 
                                                patience=3, min_delta=0.0001)])


    def predict(self, x_teste): 
        x_teste = sequence.pad_sequences(x_teste, maxlen=self.max_review_length)

        results = self.model.predict(x_teste)

        resultados = []

        for item in results:
            resultados.append(np.argmax(item))

        return resultados
