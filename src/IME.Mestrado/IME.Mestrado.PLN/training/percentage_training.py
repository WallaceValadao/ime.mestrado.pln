from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

import configs as ref_config

class PercentageTraining():
    previsores = []
    classe = []
    random_state = 0
    nameAlgoritm = ''

    X_treinamento = []
    X_teste = []
    y_treinamento = []
    y_teste = []
    taxa_acerto = 0
    algoritm = []

    def __init__(self, previsores, classe, positions, algoritm, nameAlgoritm, typeRv):
        self.configs = ref_config.Configs()

        self.previsores = previsores
        self.classe = classe
        self.positions = positions
        self.algoritm = algoritm
        self.nameAlgoritm = nameAlgoritm
        self.typeRv = typeRv

    def execute(self, maxReviewLength):
        self.part()

        self.algoritm.fit(self.X_treinamento, self.y_treinamento, maxReviewLength)

        resultados = self.algoritm.predict(self.X_teste)

        self.taxa_acerto = accuracy_score(self.y_teste, resultados)

    def part(self):
        arrayPrev = np.array_split(self.previsores, 10)
        arrayClass = np.array_split(self.classe, 10)

        for i in range(0, 10):
            if i in self.positions:
                if (len(self.X_teste) == 0):
                    self.X_teste = arrayPrev[i]
                    self.y_teste = arrayClass[i]
                else:
                    self.X_teste = [*self.X_teste, *arrayPrev[i]]
                    self.y_teste = [*self.y_teste, *arrayClass[i]]
            else:
                if (len(self.X_treinamento) == 0):
                    self.X_treinamento = arrayPrev[i]
                    self.y_treinamento = arrayClass[i]
                else:
                    self.X_treinamento = [*self.X_treinamento, *arrayPrev[i]]
                    self.y_treinamento = [*self.y_treinamento, *arrayClass[i]]

        self.X_teste = np.array(self.X_teste)
        self.y_teste = np.array(self.y_teste)
        self.X_treinamento = np.array(self.X_treinamento)
        self.y_treinamento = np.array(self.y_treinamento)

    def printResult(self):
        self.configs.log.print(self.nameAlgoritm + ' - ' + self.typeRv + ' obteve o melhor resultado: ' +  str(self.taxa_acerto))
        self.configs.log.print('Dataset treinamento nas posições:' + ''.join(map(str, self.positions)))