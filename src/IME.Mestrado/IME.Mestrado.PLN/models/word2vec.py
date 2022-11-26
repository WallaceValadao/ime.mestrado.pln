import numpy as np
from gensim.models import Word2Vec
from os.path import exists
import models.util.valorPositivo as valorPositivo
import models.util.tratarCorteFrases as tratarFrases

class PreProcessamentoW2v():
    def __init__(self, config, configModel):
        self.config = config
        self.configModel = configModel

        if not exists(configModel.pathModel):
            print(f'Building RV model from corpus {configModel.path}')
            self._build()

        print('Creating vector representations')
        self.word_vectors = Word2Vec.load(configModel.pathModel).wv
        self.word_vectors.init_sims(replace=True)


    def w2vEmbeddings(self, frases):
        result = []

        for textBase in frases:
            for textos in tratarFrases.obterCorteFrases(self.config, textBase, self.configModel.size_vector):
                textos = [sentence.split(' ') for sentence in textos]
                textos_sum = [[0]*self.word_vectors.vector_size for sentence in textos] # sentences não é usada, mas parece que funciona como equivalente a len(textos)
                for i, sentence in enumerate(textos):
                    for word in sentence:
                        try:
                            textos_sum[i] = textos_sum[i] + self.word_vectors[word]
                        except:
                            continue
                    textos_sum[i] = np.array(textos_sum[i])/len(sentence)
            result.append(textos_sum)
    
        result = valorPositivo.converterArray(result)
        return list(result)

    def w2vEmbeddings2(self, textos):
        textosPalavras = [sentence.split(' ') for sentence in textos]
        textos_sum = [[0]*self.word_vectors.vector_size for sentence in textosPalavras] # sentences não é usada, mas parece que funciona como equivalente a len(textos)
        
        result = []
        for i, sentence in enumerate(textos):
            resultText = []

            for text in tratarFrases.obterCorteFrases(self.config, sentence, self.configModel.size_vector):
                for word in text.split(' '):
                    try:
                        resultText.append(textos_sum[i] + self.word_vectors[word]);
                    except:
                        continue
                #textos_sum[i] = np.array(textos_sum[i])/len(sentence)

            result.append(resultText)
    
        result = valorPositivo.converterArray(result)
        return list(result)
    
    
    ## baseado no código de LuizGFerreira em https://github.com/Luizgferreira/subjectivity-classifier
    def _build(self):
        file_path = self.configModel.path
        sentences = np.loadtxt(file_path, dtype='str', delimiter='\t')
        sentences = [sentence.split(' ') for sentence in sentences]
        model = Word2Vec(size = self.configModel.size_vector,
                         min_count=self.configModel.min_count,
                         workers=3, window=self.configModel.window)
        model.build_vocab(sentences, progress_per=self.configModel.progress)
        model.train(sentences, total_examples=len(sentences),
                    epochs=self.configModel.epochs)
        # o nome do arquivo de modelo será o corpus_name sem a extensão
        model.save(self.configModel.pathModel)
    