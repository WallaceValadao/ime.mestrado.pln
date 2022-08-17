from spellchecker import SpellChecker
spell = SpellChecker(language = 'pt');

def correcaoOrtografica(text):
    if text == ' ':
        return '' 
    
    if not text:
        return ''

    if text.find('.') > -1:
        return text

    if text.find(',') > -1:
        return text

    if text.find('?') > -1:
        return text
        
    text = spell.correction(text)

    return text;