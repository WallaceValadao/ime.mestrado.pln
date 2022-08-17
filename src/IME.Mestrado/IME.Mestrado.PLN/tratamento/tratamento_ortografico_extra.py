import os
import json
import re

palavras_erro =     [ 'vcs', 'oq',     'aql',    'qnd',    'vc',   'nao', ' ', 'depressao', 'dvv',     'ñ',   'pq',     'mh',    'mt',     'pr',   'tadinho', 'p/',   'pqp',    'aff', 'pra',  'pfv',       'q',   'n',   'msm' ]
palavras_correcao = [ 'vocês', ' o que', 'aquilo', 'quando', 'você', 'não', '',  'depressão', 'verdade', 'não', 'porque', 'minha', 'muito', 'para', 'coitado', 'para', 'porque', 'aff', 'para', 'por favor', 'que', 'não', 'mesmo' ]

def corrigir(nome):
    if isCorrigir(nome):
        idx = palavras_erro.index(nome)

        return palavras_correcao[idx]

    return nome

def isCorrigir(nome):
    return nome in palavras_erro
