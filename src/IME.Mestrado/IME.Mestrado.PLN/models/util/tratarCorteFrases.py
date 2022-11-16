
def obterCorteFrases(config, frase, size_vector):
    if config.regraParteClassificacao == 0:
        return [ frase[0:size_vector] ]

    tamanhoFrase = len(frase)
    if size_vector > tamanhoFrase:
        return [ frase ]

    quantidde = tamanhoFrase / size_vector

    result = []

    partes = frase.split('.')
    lenPartes = len(partes)

    if lenPartes == 1:
        result.append(frase[:size_vector])
        return result

    while len(partes) > 0:
        fraseAdd = partes.pop(0)
        
        while len(fraseAdd) < size_vector and len(partes) > 0:
            tempFrase = fraseAdd + '. ' + partes[0]

            if len(tempFrase) < size_vector:
                partes.pop(0)
                fraseAdd = tempFrase
            else:
                break
        
        result.append(fraseAdd)

    return result