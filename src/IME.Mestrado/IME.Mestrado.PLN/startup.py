import pandas as pd
import os.path

import configs as config

#leitura do dataset
pathDb = config.dados.pathDb

tratarDb = True
separador = ';'
if os.path.isfile(config.dados.pathDbTratado):
    tratarDb = False
    pathDb = config.dados.pathDbTratado

dataset = pd.read_csv(pathDb, separador)
dataset.shape

previsores = dataset.iloc[:,0:1].values
classeBase = dataset.iloc[:,1].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classeBase)

#tratamento das informações
import corretor_db as corretor_db
corretorDb = corretor_db.CorretorDb()

previsores = corretorDb.getOrCorrect(tratarDb, previsores, classe)

#extração de atributos
import pre_processamento_atributos as pre_atributos
extrator = pre_atributos.ExtratorDeAtributos(previsores)

representacoes = extrator.getLiwc()

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

#simulações
import simulation_db as simulation_db
simulacoes = simulation_db.SimulationAlgorithm(algortimos, representacoes, classe)
#simulacoes.execute(220, False, True)
simulacoes.execute(0, False, True)
