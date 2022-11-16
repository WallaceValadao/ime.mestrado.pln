from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

import training.base_training as base_training

class PercentageTraining(base_training.BaseTraining):

    def __init__(self, configuracoes, previsores, classe, algoritm, positions):
        self.positions = positions
        super().__init__(configuracoes, previsores, classe, algoritm)

    def execute(self, maxReviewLength):
        self.part()

        self.convertArrayTest()

        self.algoritm.fit(self.X_treinamento, self.y_treinamento, maxReviewLength)

        self.calcularAcerto()

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