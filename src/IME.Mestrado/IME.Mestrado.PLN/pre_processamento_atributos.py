from pandas.core.base import DataError
import models.liwc_all as liwcHelper
import models.liwc_stopwords as liwcHelperStopwords
import models.bert as bertImport
import models.word2vec as w2vimport ######################
import configs as config
import os.path
import pandas as pd
import codecs, json 

class Rv(object):
    def __init__(self, name, dataset, max_review_length, execucao):
        self.name = name
        
        self.max_review_length = max_review_length
        self.pathRv = f'{config.dados.pathRepresentacao}{self.name}.json';

        if os.path.isfile(self.pathRv):
            obj_text = codecs.open(self.pathRv, 'r', encoding='utf-8').read()
            b_new = json.loads(obj_text)
            #self.previsores = np.array(b_new)
            self.previsores = b_new
        else:
            self.previsores = execucao.processar(dataset)

            listJson = self.previsores

            json.dump(listJson, codecs.open(self.pathRv, 'w', encoding='utf-8'), 
                            separators=(',', ':'), sort_keys=True, indent=4)


    def getName(self):
        return self.name

    def getPrevisores(self):
        return self.previsores

    def getMaxReviewLength(self):
        return self.max_review_length

class GetLiwc():
    def __init__(self, liwc):
        self.liwc = liwc


    def processar(self, dataset):
        previsoresLiwc = []

        for text in dataset:
            previsoresLiwc.append(self.liwc.getAttributes(text))

        return previsoresLiwc

class GetBert():
    def __init__(self, bpath, max_len):
        self.bpath = bpath
        self.max_len = max_len

    def processar(self, dataset):
        bert = bertImport.PreProcessamentoBert(self.bpath, self.max_len, True)

        return bert.getAttributesBase(dataset)
        #return bert.preprocessing_for_bert(dataset)

class GetWord2Vec():
    def __init__(self, model):
        self.model = model

    def processar(self, dataset):
        w2v = w2vimport.PreProcessamentoW2v(self.model)

        return w2v.w2vEmbeddings(dataset)

class ExtratorDeAtributos():

    def __init__(self, previsores):
        self.previsores = previsores

        pathLiwc = 'asserts\\modelos\\LIWC2007_Portugues_win.dic.json'

        self.liwc = liwcHelper.initLiwc(pathLiwc)
        self.liwcStopWords = liwcHelperStopwords.initLiwc(pathLiwc)

        self.liwcFilter = liwcHelper.initLiwc(pathLiwc, ['posemo', 'negemo'])
        self.liwcStopWordsFilter = liwcHelperStopwords.initLiwc(pathLiwc, ['posemo', 'negemo'])

        self.liwcPosemoFilter = liwcHelper.initLiwc(pathLiwc, ['posemo'])
        self.liwcPosemoStopWordsFilter = liwcHelperStopwords.initLiwc(pathLiwc, ['posemo'])

        self.liwcNegemoFilter = liwcHelper.initLiwc(pathLiwc, ['negemo'])
        self.liwcNegemoStopWordsFilter = liwcHelperStopwords.initLiwc(pathLiwc, ['negemo'])


    def getLiwc(self):
        list = []

        list.append(Rv('Liwc', self.previsores, 64, GetLiwc(self.liwc)))
        list.append(Rv('Liwc_(com_stopwords)', self.previsores, 64, GetLiwc(self.liwcStopWords)))
        
        list.append(Rv('Liwc_posemo_negemo', self.previsores, 2, GetLiwc(self.liwcFilter)))
        list.append(Rv('Liwc_posemo_negemo_(com_stopwords)', self.previsores, 2, GetLiwc(self.liwcStopWordsFilter)))
        
        list.append(Rv('Liwc_posemo', self.previsores, 1, GetLiwc(self.liwcPosemoFilter)))
        list.append(Rv('Liwc_posemo_(com_stopwords)', self.previsores, 1, GetLiwc(self.liwcPosemoStopWordsFilter)))
        
        list.append(Rv('Liwc_negemo', self.previsores, 1, GetLiwc(self.liwcNegemoFilter)))
        list.append(Rv('Liwc_negemo_(com_stopwords)', self.previsores, 1, GetLiwc(self.liwcNegemoStopWordsFilter)))

        return list

    def getBert(self):
        list = []

        for bpath in config.dados.bert_array:
            list.append(Rv(f'{bpath.nome}_768', self.previsores, 768, GetBert(bpath.path, 768)))
            #list.append(Rv('Bert_512', self.previsores, 512, GetBert(512)))
            #list.append(Rv('Bert_256', self.previsores, 256, GetBert(256)))
            #list.append(Rv('Bert_64', self.previsores, 64, GetBert(64)))

        return list
    # o modelCorpus deverá ser o nome do arquivo do corpus da RV, não o da rede neural.

    def getWord2Vec(self):
        list = []

        for w2v in config.dados.w2VEmbeddings:
            list.append(Rv(f'Word2Vec_{w2v.size_vector}', 
                           self.previsores, 768, GetWord2Vec(w2v)))

        return list