import pandas as pd
import os.path

import configs as config


#leitura do dataset
pathDb = config.dados.pathDb

tratarDb = False
separador = ','
if os.path.isfile(config.dados.pathDbTratado):
    tratarDb = False
    pathDb = config.dados.pathDbTratado
print(f'Starting training for the dataset in {pathDb}')
dataset = pd.read_csv(pathDb, separador)
print(dataset.shape)

previsores = dataset.iloc[:,[7, 4]].values
print(previsores[0:5])
classeBase = dataset.iloc[:,4].values
print(classeBase[0:5])

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classeBase)

#tratamento das informa��es
import corretor_db as corretor_db
corretorDb = corretor_db.CorretorDb()

previsores = corretorDb.getOrCorrect(tratarDb, previsores, classe)

#extra��o de atributos
import pre_processamento_atributos as pre_atributos
extrator = pre_atributos.ExtratorDeAtributos(previsores)

#representacoes = extrator.getLiwc()

representacoes = extrator.getWord2Vec() ######################EM CONSTRUÇÃO###############

#representacoes += extrator.getBert()
#representacoes = extrator.getBert()

#criando lista de algortimos
import algoritmos as algoritmos

algortimos = []
listAlgoritmos = algoritmos.AlgoritmosList()

#uso algoritmos classicos
classics = listAlgoritmos.getClassic()
algortimos = classics

#algortimos de redes neurais
#rna = listAlgoritmos.getRna()

#algortimos += rna
#algortimos = rna

#simula��es
import simulation_db as simulation_db
simulacoes = simulation_db.SimulationAlgorithm(algortimos, representacoes, classe)
#simulacoes.execute(220, False, True)
simulacoes.execute(0, False, True)
