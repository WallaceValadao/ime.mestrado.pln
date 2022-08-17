import os
import re

def _removerDuplicados(texto, caract, caractDuple):
    while texto.find(caractDuple) > -1:
        texto = texto.replace(caractDuple, caract)

    return texto;

def removeDuplicado(texto):
    texto = _removerDuplicados(texto, 'k', 'kk')
    texto = _removerDuplicados(texto, '.', '.  .')
    texto = _removerDuplicados(texto, '.', '. .')
    texto = _removerDuplicados(texto, '.', '..')
    texto = _removerDuplicados(texto, ',', ',,')
    texto = _removerDuplicados(texto, '??', '? ?')
    texto = _removerDuplicados(texto, ' ', '  ')
    texto = _removerDuplicados(texto, '?', '??')
    texto = _removerDuplicados(texto, ' k ', ' risos ')

    return texto