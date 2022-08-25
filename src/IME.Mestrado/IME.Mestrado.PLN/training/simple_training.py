from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import configs as ref_config

class SimpleTraining():
    previsores = []
    classe = []
    test_size = 0.3
    random_state = 0
    nameAlgoritm = ''

    X_treinamento = []
    X_teste = []
    y_treinamento = []
    y_teste = []
    taxa_acerto = 0
    algoritm = []

    def __init__(self, previsores, classe, test_size, random_state, algoritm, nameAlgoritm, typeRv):
        self.configs = ref_config.Configs()

        self.previsores = previsores
        self.classe = classe
        self.test_size = test_size
        self.random_state = random_state
        self.algoritm = algoritm
        self.nameAlgoritm = nameAlgoritm
        self.typeRv = typeRv

    def execute(self, maxReviewLength):
        self.X_treinamento, self.X_teste, self.y_treinamento, self.y_teste = train_test_split(self.previsores,
                                                                  self.classe,
                                                                  test_size = self.test_size,
                                                                  random_state = self.random_state)

        self.algoritm.fit(self.X_treinamento, self.y_treinamento, maxReviewLength)

        resultados = self.algoritm.predict(self.X_teste)

        self.taxa_acerto = accuracy_score(self.y_teste, resultados)

    def printResult(self):
        self.configs.log.print(self.nameAlgoritm + ' - ' + self.typeRv + ' - obteve o melhor resultado: ' +  str(self.taxa_acerto))
        self.configs.log.print('Dataset treinamento com ' + str(self.test_size * 100) + '% para teste')