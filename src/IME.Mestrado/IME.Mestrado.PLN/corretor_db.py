
class CorretorDb():
    
    def getOrCorrect(self, configuracoes, tratarDb, previsores, classe):
        if tratarDb:
            import pre_processamento_base as preProcessamento
            pre = preProcessamento.PreProcessamentoBase()
            previsores = pre.execute(previsores)
            
            qt = len(previsores)
        
            import csv
        
            with open(configuracoes.pathDbTratado, 'w', newline='', encoding='utf-8') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter=';')
        
                for i in range(0, qt):
                    wr.writerow([previsores[i], classe[i]])
            return previsores

        novaPrevi = []
        for previ in previsores:
            novaPrevi.append(previ[0])
        # print(novaPrevi) para debug
        return novaPrevi
        