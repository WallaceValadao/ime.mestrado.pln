
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

    def __init__(self):
        #caminho para arquivo com o dataset
        self.pathDb = 'datasets\\Computer-BR-preproc.csv'
        self.pathDbTratado = self.pathDb.replace('.csv', '_tratado.csv')
        self.log = printFile
        self.numero_classes = 2

        #Aqui pode definir a quantidade de epochs para as rnas
        self.epochs = [ 2, 10, 20, 50 ]
        #self.epochs = [ 2 ]


dados = Configs()


