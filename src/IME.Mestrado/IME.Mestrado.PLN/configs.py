import os

class PrintBase():
    
    def print(self, text):
        print(text)


    def save(self):
        print('Finalizado..')


class PrintFile():

    def __init__(self, dataset):
        from time import gmtime, strftime
        data = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

        self.textSave = ''
        self.pathFile = f'asserts\\resultados\\{dataset}-{data}.txt'

    
    def print(self, text):
        print(text)

        self.textSave += text
        self.textSave += '\n'

    def save(self):
        print('Finalizado..')

        with open(self.pathFile, 'w') as f:
            f.write(self.textSave)


class W2VModel():
    def __init__(self, path, pathModel, size_vector, min_count, window, epochs, progress):
        self.path = path
        self.pathModel = pathModel
        self.size_vector = size_vector
        self.min_count = min_count
        self.window = window
        self.epochs = epochs
        self.progress = progress

class BertModel():
    def __init__(self, nome, path):
        self.nome = nome
        self.path = path


class Configs():
    
    def __init__(self, datasetName):
        self.pathLiwc = 'asserts\\modelos\\LIWC2007_Portugues_win.dic.json'

        #caminho para arquivo com o dataset
        dataset = datasetName.split('.')[0]

        self.pathDb = f'asserts\\datasets\\{datasetName}'
        self.pathDbTratado = f'asserts\\datasets_tratados\\{datasetName}'
        self.mediaPath = 'asserts\\resultados\\medias.json'
        self.separador = ';'
        #self.dfColumns = {
        #    'text': 'FRASE',
        #    'classes': 'OBJ/SUBJ'
        #}
        self.dfColumns = {
            'text': 'frases',
            'classes': 'classes'
        }
        self.tratarDb = False
        self.pathRepresentacao = f'asserts\\rv_dataset\\{dataset}_'
        self.log = PrintFile(dataset)
        self.numero_classes = 2
        self.w2VEmbeddings = [
            W2VModel('asserts\\modelos\\buscape_preprocessed.txt', 
                     'asserts\\rv_models\\w2v-buscape_preprocessed_768.model', 
                     768, 4, 4, 100, 10000)
        ]
        self.bert_array = [
            BertModel('Bertimbau_base', 'neuralmind/bert-base-portuguese-cased'),
            BertModel('Bertimbau_large', 'neuralmind/bert-large-portuguese-cased'),
            BertModel('Bert_base', 'bert-base-uncased'),
            BertModel('Bert_large', 'bert-large-uncased'),
            BertModel('Bert_pierreguillou', 'pierreguillou/bert-base-cased-squad-v1.1-portuguese'),
            BertModel('Bert_multilingual', 'bert-base-multilingual-cased'),
            BertModel('Roberta_xlm_base', 'xlm-roberta-base'),
            BertModel('Roberta_xlm_large', 'xlm-roberta-large'),
            BertModel('Roberta_josu', 'josu/roberta-pt-br'),

            #https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment
            BertModel('Roberta_cardiffnlp', 'cardiffnlp/twitter-xlm-roberta-base'),
            BertModel('Roberta_cardiffnlp_sentiment', 'cardiffnlp/twitter-xlm-roberta-base-sentiment'),
        ]
        #Aqui pode definir a quantidade de epochs para as rnas
        self.epochs = [2, 10, 20, 50]
        #self.epochs = [ 2 ]


array_configuracoes = []

for namePathDb in os.listdir('asserts\\datasets'):
    array_configuracoes.append(Configs(namePathDb))


