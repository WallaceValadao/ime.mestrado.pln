from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class BaseTraining():
    previsores = []
    classe = []
    nameAlgoritm = ''

    X_treinamento = []
    X_teste = []
    y_treinamento = []
    y_teste = []
    taxa_acerto = 0
    algoritm = []

    def __init__(self, configuracoes, previsores, classe, algoritm):
        self.configs = configuracoes

        self.previsores = previsores.getPrevisores()
        self.classe = classe
        self.algoritm = algoritm.getInstance()
        self.nameAlgoritm = algoritm.getName()
        self.typeRv = previsores.getName()

    def convertArrayTest(self):
        arrayX_treinamento = []
        arrayY_treinamento = []

        for i in range(0, len(self.X_treinamento)):
            for text in self.X_treinamento[i]:
                arrayX_treinamento.append(text)
                arrayY_treinamento.append(self.y_treinamento[i])

        self.X_treinamento = arrayX_treinamento
        self.y_treinamento = arrayY_treinamento

    def calcularAcerto(self):
        resultados = []

        for item in self.X_teste:
            result = self.algoritm.predict(item)

            if self.configs.regraParteClassificacao == 1 and self.configs.valorParteClassificador in result:
                resultados.append(self.configs.valorParteClassificador)
            else:
                resultados.append(result[0])

        self.taxa_acerto = accuracy_score(self.y_teste, resultados)