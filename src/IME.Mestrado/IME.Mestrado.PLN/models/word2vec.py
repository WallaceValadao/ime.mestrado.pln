
import numpy as np
from gensim.models import Word2Vec
from os.path import exists
import configs as config


def w2vEmbeddings(textos):
    # # caso seja usado um novo conjunto dataset-dimensão para construir os embeddigns
    if not exists(f'w2v-{config.dados.w2vCorpus[9:-4]} {config.dados.W2VEmbeddings["size_vector"]}.model'):
        print(f'Building RV model from corpus {config.dados.w2vCorpus[9:]}')
        build(config.dados.w2vCorpus)
    print('Creating vector representations')
    word_vectors = Word2Vec.load(f'w2v-{config.dados.w2vCorpus[9:-4]} {config.dados.W2VEmbeddings["size_vector"]}.model').wv
    word_vectors.init_sims(replace=True)
    textos = [sentence.split(' ') for sentence in textos]
    textos_sum = [[0]*word_vectors.vector_size for sentence in textos] # sentences não é usada, mas parece que funciona como equivalente a len(textos)
    for i, sentence in enumerate(textos):
        for word in sentence:
            try:
                textos_sum[i] = textos_sum[i] + word_vectors[word]
            except:
                continue
        textos_sum[i] = np.array(textos_sum[i])/len(sentence)

    parsePositive(textos_sum)
    return np.array(textos_sum)


## baseado no código de LuizGFerreira em https://github.com/Luizgferreira/subjectivity-classifier
def build(corpus):
    file_path = corpus
    sentences = np.loadtxt(file_path, dtype='str', delimiter='\t')
    sentences = [sentence.split(' ') for sentence in sentences]
    model = Word2Vec(vector_size = config.dados.W2VEmbeddings['size_vector'],
                     min_count=config.dados.W2VEmbeddings['min_count'],
                     workers=3, window=config.dados.W2VEmbeddings['window'])
    model.build_vocab(sentences, progress_per=config.dados.W2VEmbeddings['progress'])
    model.train(sentences, total_examples=len(sentences),
                epochs=config.dados.W2VEmbeddings['epochs'])
    # o nome do arquivo de modelo será o corpus_name sem a extensão
    model.save(f'w2v-{corpus[9:-4]} {config.dados.W2VEmbeddings["size_vector"]}.model')

def parsePositive(list):
    minUser = 0

    for item in list:
        for atual in item:
            if atual < minUser:
                minUser = atual

    minUser = minUser * -1

    for item in list:
        for i in range(0, len(item)):
            item[i] += minUser

    return list