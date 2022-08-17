import os
import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text) 
    text = BAD_SYMBOLS_RE.sub('', text) 
#    text = re.sub(r'\W+', '', text)
    return text


def removeCaracteres(texto):
    texto = clean_text(texto)

    texto = texto.replace('\n', '. ')
    texto = texto.replace('!', '.')
    texto = texto.replace(',', ' , ')
    texto = texto.replace('?', ' ? ')
    texto = texto.replace('"', ' ')
    texto = texto.replace('.', ' . ')
    texto = texto.replace('/', ' ')
    texto = texto.replace('\\', ' ')
    #texto = texto.replace('“', '')
    #texto = texto.replace('”', '')

    return texto
