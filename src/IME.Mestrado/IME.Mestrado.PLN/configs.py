
class PrintBase():
    
    def print(self, text):
        print(text)


    def save(self):
        print('Finalizado..')


class PrintFile():

    def __init__(self):
        from time import gmtime, strftime
        data = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

        self.textSave = ''
        self.pathFile = 'results\\resultados-' + data + '.txt'

    
    def print(self, text):
        print(text)

        self.textSave += text
        self.textSave += '\n'

    def save(self):
        print('Finalizado..')

        with open(self.pathFile, 'w') as f:
            f.write(self.textSave)

printBase = PrintBase()
printFile = PrintFile()

class Configs():

    def __init__(self, datasetName, datasetPath, w2vCorpus):
        #caminho para arquivo com o dataset
        self.dataset = datasetName
        self.pathDb = f'datasets\\{datasetPath}'
        self.separador = ','
        self.dfColumns = {
            'text': 'FRASE',
            'classes': 'OBJ/SUBJ'
        }
        self.tratarDb = False
        self.pathDbTratado = self.pathDb.replace('.csv', '_tratado.csv')
        self.log = printFile
        self.numero_classes = 2
        self.w2vCorpus = f'datasets\\{w2vCorpus}'
        self.W2VEmbeddings = {
            'size_vector' : 768,
            'min_count' : 4,
            'window' : 4,
            'epochs' : 100,
            'progress' : 10000
        }
        #Aqui pode definir a quantidade de epochs para as rnas
        self.epochs = [2, 10, 20, 50]
        #self.epochs = [ 2 ]

### Modelo a ser construído para usar de entrada
w2vCorpora = {
    'buscape': 'buscape_preprocessed.txt'
}

### Dataset a ser treinado com o modelo
datasets = {
    'ComputerBR': 'Computer-BR-preproc.csv',
    'book reviews': 'corpus_book_reviews_portuguese_preproc.csv',
    'electronic reviews': 'Subjectivity-annotated_corpus_on_electronic_product_domain-anotacao-BELISARIO-preproc.CSV'

}
### Escolha, conforme conveniente, o nome do dataset de treinamento e o corpus de treinamento do word2vec.
datasetName = 'electronic reviews'
w2vCorpus = 'buscape'


dados = Configs(
    datasetName,
    datasets[datasetName],
    w2vCorpora[w2vCorpus]
)


