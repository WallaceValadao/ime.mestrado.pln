import pandas as pd
import os.path
from sklearn.preprocessing import LabelEncoder
import pre_processamento_atributos as pre_atributos
import algoritmos as algoritmos
import simulation_db as simulation_db

import configs as config

for configuracoes in config.array_configuracoes:
    #leitura do dataset
    pathDb = configuracoes.pathDb
    
    tratarDb = configuracoes.tratarDb
    separador = configuracoes.separador
    if os.path.isfile(configuracoes.pathDbTratado):
        tratarDb = False
        pathDb = configuracoes.pathDbTratado
    print(f'Starting training for the dataset in {pathDb}')
    dataset = pd.read_csv(pathDb, separador, skipinitialspace=configuracoes.skipinitialspace)
    print(dataset.shape)
    textCol = configuracoes.dfColumns['text']
    classCol = configuracoes.dfColumns['classes']
    
    posicaoText = 0
    posicaoClasses = 0
    
    for posicao in range(len(dataset.columns.values)):
        if textCol == dataset.columns.values[posicao]:
            posicaoText = posicao
        elif classCol == dataset.columns.values[posicao]:
            posicaoClasses = posicao
    
    previsores = dataset.iloc[:,[posicaoText, posicaoClasses]].values
    print(previsores[0:5])
    classeBase = dataset.iloc[:,posicaoClasses].values
    print(classeBase[0:5])
    
    labelencoder = LabelEncoder()
    classe = labelencoder.fit_transform(classeBase)
    
    #tratamento das informações
    import corretor_db as corretor_db
    corretorDb = corretor_db.CorretorDb()
    
    previsores = corretorDb.getOrCorrect(configuracoes, tratarDb, previsores, classe)
    
    #extra��o de atributos
    extrator = pre_atributos.ExtratorDeAtributos(configuracoes, previsores)
    
    #representacoes = extrator.getLiwc()
    representacoes = extrator.getWord2Vec()
    #representacoes = extrator.getBert()
    
    #criando lista de algortimos
    algortimos = []
    listAlgoritmos = algoritmos.AlgoritmosList(configuracoes)
    
    #uso algoritmos classicos
    algortimos = listAlgoritmos.getClassic()
    
    #algortimos de redes neurais
    rna = listAlgoritmos.getRna()
    
    #algortimos += rna
    algortimos = rna
    
    #simula��es
    simulacoes = simulation_db.SimulationAlgorithm(configuracoes, algortimos, representacoes, classe)
    simulacoes.execute()
