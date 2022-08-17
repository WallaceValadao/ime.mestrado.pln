import os
import json
import re

import tratamento.tratamento_ortografico_extra as corretor1
import tratamento.corretor_ortografico as corretor2
import tratamento.remover_emojify as emojify
import tratamento.duplicados as duplicados
import tratamento.remover_caracteres as caracteres

class PreProcessamentoBase():

    def __init__(self):
        self.dicresoucer = []
        self.dicresoucer.append(emojify.removeEmojify)
        self.dicresoucer.append(corretor1.corrigir)
        self.dicresoucer.append(corretor2.correcaoOrtografica)
    
    def execute(self, previsores):
        previsoresCorrigido = []

        for text in previsores:
            novaFrase = text[0].lower()
            novaFraseSemDuplicados = duplicados.removeDuplicado(novaFrase)
            novaFraseSemDuplicados = caracteres.removeCaracteres(novaFraseSemDuplicados)

            palavras = novaFraseSemDuplicados.split(' ')
            qt = len(palavras)
       
            for i in range(0, qt):
                for executer in self.dicresoucer:
                    palavras[i] = executer(palavras[i])
        
            previsoresCorrigido.append(' '.join(palavras))

        return previsoresCorrigido