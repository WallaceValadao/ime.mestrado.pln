from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier

import redes_neurais.lstm as lstm
import redes_neurais.bilstm as bilstm

import configs as ref_config

class Algoritmo():

    def __init__(self, name, instance):
        self.name = name
        self.instance = instance

    def getName(self):
        return self.name

    def getInstance(self):
        return self.instance
    

class MachineLearningClassification():

    def __init__(self, classication):
        self.classication = classication

    def _createInstance(self):
        self.model = self.classication()


    def fit(self, X_train, Y_train, maxReviewLength):
        self._createInstance()

        self.model.fit(X_train, Y_train)


    def predict(self, x_teste): 
        results = self.model.predict(x_teste)
        
        return results


class AlgoritmosList():

    def __init__(self):
        self.configs = ref_config.Configs()


    def getClassic(self):
        algoritmosClassic = []

        #algoritmosClassic.append(Algoritmo('Gaussian Naive Bayes', MachineLearningClassification(GaussianNB)))
        algoritmosClassic.append(Algoritmo('Multinomial Naive Bayes', MachineLearningClassification(MultinomialNB)))
        algoritmosClassic.append(Algoritmo('Bernoulli Naive Bayes', MachineLearningClassification(BernoulliNB)))
        algoritmosClassic.append(Algoritmo('Complement Naive Bayes (CNB)', MachineLearningClassification(ComplementNB)))

        algoritmosClassic.append(Algoritmo('Decision tree classifier', MachineLearningClassification(DecisionTreeClassifier)))
        algoritmosClassic.append(Algoritmo('Support Vector Classification (SVC)', MachineLearningClassification(SVC)))
        
        algoritmosClassic.append(Algoritmo('random forest classifier', MachineLearningClassification(RandomForestClassifier)))
        
        algoritmosClassic.append(Algoritmo('Extra-trees classifier', MachineLearningClassification(ExtraTreesClassifier)))

        algoritmosClassic.append(Algoritmo('AdaBoost', MachineLearningClassification(AdaBoostClassifier)))
        algoritmosClassic.append(Algoritmo('Gradient Boosting Machine', MachineLearningClassification(GradientBoostingClassifier)))

        #algoritmosClassic.append(Algoritmo('k-nearest neighbors (KNN) n_neighbors = 2', KNeighborsClassifier(n_neighbors = 2)))
        #algoritmosClassic.append(Algoritmo('k-nearest neighbors (KNN) n_neighbors = 2', KNeighborsClassifier(n_neighbors = 2)))
        #algoritmosClassic.append(Algoritmo('k-nearest neighbors (KNN) n_neighbors = 3', KNeighborsClassifier(n_neighbors = 3)))
        #algoritmosClassic.append(Algoritmo('k-nearest neighbors (KNN) n_neighbors = 5', KNeighborsClassifier(n_neighbors = 5)))
        #algoritmosClassic.append(Algoritmo('random forest classifier n_estimators = 50', RandomForestClassifier(n_estimators = 50)))
        #algoritmosClassic.append(Algoritmo('random forest classifier n_estimators = 100', RandomForestClassifier(n_estimators = 100)))
        #algoritmosClassic.append(Algoritmo('random forest classifier n_estimators = 200', RandomForestClassifier(n_estimators = 200)))
        #algoritmosClassic.append(Algoritmo('Histogram-Based Gradient Boosting', HistGradientBoostingClassifier()))
        #algoritmosClassic.append(Algoritmo('Extra-trees classifier', ExtraTreesClassifier(n_estimators = 200)))

        return algoritmosClassic

    def getRna(self):
        algoritmosRna = []

        for epoch in self.configs.epochs:
            labelEpoch =  ' (epoch = ' + str(epoch) + ')'
            algoritmosRna.append(Algoritmo('LSTM (softmax)' + labelEpoch, lstm.LstmClassification('softmax', epoch)))
            algoritmosRna.append(Algoritmo('LSTM (sigmoid)' + labelEpoch, lstm.LstmClassification('sigmoid', epoch)))
            algoritmosRna.append(Algoritmo('BiLSTM (softmax)' + labelEpoch, bilstm.BiLstmClassification('softmax', epoch)))
            algoritmosRna.append(Algoritmo('BiLSTM (sigmoid)' + labelEpoch, bilstm.BiLstmClassification('sigmoid', epoch)))

        return algoritmosRna