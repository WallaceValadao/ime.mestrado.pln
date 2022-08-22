import pandas as pd
import training.simple_training as training;
import training.percentage_training as percentageTraining;

import statistics
import configs as ref_config

class Object(object):
    pass

class SimulationAlgorithm():
    previsores = []
    classe = []
    bestResult = 0
    bestAlgorithm = ''

    algoritmos = []
    algoritmosLabel = []

    def __init__(self, algoritmos, previsores, classe):
        self.configs = ref_config.Configs()

        self.previsores = previsores;
        self.classe = classe;

        self.algoritmos = algoritmos

        self.worseResults = [];
        self.besteResults = [];
        self.mediaResults = [];
        self.quantidadeRodadas = 0;

        for j in range(0, len(self.previsores)):
            for i in range(0, len(self.algoritmos)):
                nameAg = self.algoritmos[i].getName() + ' - ' + self.previsores[j].getName()

                self.worseResults.append(None)
                self.besteResults.append(None)

                mediaAcerto = Object()
                mediaAcerto.valores = []
                mediaAcerto.nome = nameAg
                self.mediaResults.append(mediaAcerto)

    def execute(self, quantidadeRodadas = 0, usarParticaoSimples = True, usarParticaoFracionado = True):
        for prev in self.previsores:
            self.quantidadeRodadas = 0
            self.executePrevisores(prev, quantidadeRodadas, usarParticaoSimples, usarParticaoFracionado)

        self.configs.log.print('Quantidade rodadas: ' + str(self.quantidadeRodadas))
        self.configs.log.print('')

        self.configs.log.print('Piores resultados')
        for i in range(0, len(self.worseResults)):
            if self.worseResults[i] == None:
                continue

            self.worseResults[i].printResult();
        self.configs.log.print('')

        self.configs.log.print('Melhores resultados')
        for i in range(0, len(self.besteResults)):
            if self.besteResults[i]  == None:
                continue

            self.besteResults[i].printResult();
        self.configs.log.print('')

        self.configs.log.print('Media resultados + desvio padrão')
        for i in range(0, len(self.mediaResults)):
            if len(self.mediaResults[i].valores) == 0:
                continue

            media = str(statistics.mean(self.mediaResults[i].valores))
            desvio = str(statistics.pstdev(self.mediaResults[i].valores))
            self.configs.log.print(self.mediaResults[i].nome + ' ' + media + ' (' + desvio + ')')
        self.configs.log.print('')
        
        self.configs.log.print('')
        print('Melhor resultado')
        self.bestAlgorithm.printResult();

        self.configs.log.save()


    def executePrevisores(self, previsor, quantidadeRodadas, usarParticaoSimples, usarParticaoFracionado):
        if usarParticaoSimples:
            self.configs.log.print('Particionamento simples')
            self.configs.log.print('30% teste')
            self.executeSimple(0.3, previsor);
            self.configs.log.print('')
            self.configs.log.print('50% teste')
            self.executeSimple(0.5, previsor);
            self.configs.log.print('')
            self.configs.log.print('')

            self.quantidadeRodadas = 2

        if usarParticaoFracionado == False:
            return

        self.configs.log.print('Particionamento fracionado')

        for i in range(0, 10):
            self.configs.log.print('10% teste - posicões: i=' + str(i))
            self.executePart([i], previsor)
            self.configs.log.print('')

            self.quantidadeRodadas += 1

            if quantidadeRodadas > 0 and quantidadeRodadas < self.quantidadeRodadas:
                break
                    

    def executeSimple(self, percente, previsores):
        for i in range(0, len(self.algoritmos)):
            execution = training.SimpleTraining(previsores.getPrevisores(), self.classe, percente, 0, self.algoritmos[i].getInstance(), self.algoritmos[i].getName(), previsores.getName())
            execution.execute(previsores.getMaxReviewLength())
            self.validate(execution, previsores)

    def executePart(self, positions, previsores):
        for i in range(0, len(self.algoritmos)):
            nbGaussian = percentageTraining.PercentageTraining(previsores.getPrevisores(), self.classe, positions, self.algoritmos[i].getInstance(), self.algoritmos[i].getName(), previsores.getName())
            nbGaussian.execute(previsores.getMaxReviewLength())
            self.validate(nbGaussian, previsores)
        
    def validate(self, algorithm, previsor): 
        nome = algorithm.nameAlgoritm + ' - ' + previsor.getName()
        self.configs.log.print(nome + ': ' + str(algorithm.taxa_acerto))

        if (algorithm.taxa_acerto > self.bestResult):
            self.bestResult = algorithm.taxa_acerto
            self.bestAlgorithm = algorithm
            algorithm.printResult()

        posicao = 0
        for i in range(0, len(self.mediaResults)):
            if self.mediaResults[i].nome == nome:
                posicao = i
                break

        self.mediaResults[posicao].valores.append(algorithm.taxa_acerto)

        if (self.worseResults[posicao] == None or algorithm.taxa_acerto < self.worseResults[i].taxa_acerto):
            self.worseResults[posicao] = algorithm

        if (self.besteResults[posicao] == None or algorithm.taxa_acerto > self.besteResults[i].taxa_acerto):
            self.besteResults[posicao] = algorithm
