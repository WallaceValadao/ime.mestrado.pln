from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import training.base_training as base_training

class SimpleTraining(base_training.BaseTraining):
    test_size = 0.3

    def __init__(self, configuracoes, previsores, classe, algoritm, test_size):
        self.test_size = test_size
        super().__init__(configuracoes, previsores, classe, algoritm)


    def execute(self, maxReviewLength):
        self.X_treinamento, self.X_teste, self.y_treinamento, self.y_teste = train_test_split(self.previsores,
                                                                  self.classe,
                                                                  test_size = self.test_size,
                                                                  random_state = self.configs.random_state)

        self.convertArrayTest()

        self.algoritm.fit(self.X_treinamento, self.y_treinamento, maxReviewLength)

        self.calcularAcerto()


    def printResult(self):
        self.configs.log.print(self.nameAlgoritm + ' - ' + self.typeRv + ' - obteve o melhor resultado: ' +  str(self.taxa_acerto))
        self.configs.log.print('Dataset treinamento com ' + str(self.test_size * 100) + '% para teste')