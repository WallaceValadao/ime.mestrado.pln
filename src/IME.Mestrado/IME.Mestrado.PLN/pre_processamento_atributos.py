import models.liwc_all as liwcHelper
import models.liwc_stopwords as liwcHelperStopwords
import models.bert as bertImport
import models.word2vec as w2vimport ######################
import configs as config

class Rv(object):
    def __init__(self, name, previsores, max_review_length):
        self.name = name
        self.previsores = previsores
        self.max_review_length = max_review_length

    def getName(self):
        return self.name

    def getPrevisores(self):
        return self.previsores

    def getMaxReviewLength(self):
        return self.max_review_length

class ExtratorDeAtributos():

    def __init__(self, previsores):
        self.previsores = previsores

        pathLiwc = 'LIWC2007_Portugues_win.dic.json'

        self.liwc = liwcHelper.initLiwc(pathLiwc)
        self.liwcStopWords = liwcHelperStopwords.initLiwc(pathLiwc)

        self.liwcFilter = liwcHelper.initLiwc(pathLiwc, ['posemo', 'negemo'])
        self.liwcStopWordsFilter = liwcHelperStopwords.initLiwc(pathLiwc, ['posemo', 'negemo'])

        self.liwcPosemoFilter = liwcHelper.initLiwc(pathLiwc, ['posemo'])
        self.liwcPosemoStopWordsFilter = liwcHelperStopwords.initLiwc(pathLiwc, ['posemo'])

        self.liwcNegemoFilter = liwcHelper.initLiwc(pathLiwc, ['negemo'])
        self.liwcNegemoStopWordsFilter = liwcHelperStopwords.initLiwc(pathLiwc, ['negemo'])
    
    def _getLiwc(self, liwc):
        previsoresLiwc = []

        for text in self.previsores:
            previsoresLiwc.append(liwc.getAttributes(text))

        return previsoresLiwc
    
    def _getBert(self, max_len):
        bert = bertImport.PreProcessamentoBert(max_len, True)
        return bert.getAttributesBase(self.previsores)
        #return bert.preprocessing_for_bert(self.previsores)

    def _getWord2Vec(self):####################################
        return w2vimport.w2vEmbeddings(self.previsores)

    def getLiwc(self):
        list = []

        list.append(Rv('Liwc', self._getLiwc(self.liwc), 64))
        list.append(Rv('Liwc (com stopwords)', self._getLiwc(self.liwcStopWords), 64))
        
        list.append(Rv('Liwc posemo e negemo', self._getLiwc(self.liwcFilter), 2))
        list.append(Rv('Liwc posemo e negemo (com stopwords)', self._getLiwc(self.liwcStopWordsFilter), 2))
        
        list.append(Rv('Liwc posemo', self._getLiwc(self.liwcPosemoFilter), 1))
        list.append(Rv('Liwc posemo (com stopwords)', self._getLiwc(self.liwcPosemoStopWordsFilter), 1))
        
        list.append(Rv('Liwc negemo', self._getLiwc(self.liwcNegemoFilter), 1))
        list.append(Rv('Liwc negemo (com stopwords)', self._getLiwc(self.liwcNegemoStopWordsFilter), 1))

        return list

    def getBert(self):
        list = []

        list.append(Rv('Bert 768', self._getBert(768), 768))
        #list.append(Rv('Bert 512', self._getBert(512), 512))
        #list.append(Rv('Bert 160', self._getBert(160), 160))
        #list.append(Rv('Bert 256', self._getBert(256), 256))
        #list.append(Rv('Bert 64', self._getBert(64), 64))

        return list
    # o modelCorpus deverá ser o nome do arquivo do corpus da RV, não o da rede neural.
    def getWord2Vec(self):  # #####################################
        list = []
        list.append(Rv(f'Word2Vec {config.dados.dataset} {config.dados.W2VEmbeddings["size_vector"]}', self._getWord2Vec(), 768))  # max review length??

        return list