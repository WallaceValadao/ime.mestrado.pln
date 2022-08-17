import os
import json

class Liwc(object):
    liwc = []
    classes = []

    def __init__(self, liwc, filtro):
        self.liwc = liwc
        self.filtro = filtro

        for li in liwc.keys():
            for prop in liwc[li]:
                if prop in self.classes:
                    continue
                
                self.classes.append(prop)


    def getFilter(self, result):
        pfilter = []

        for i in range(0, len(self.classes)):
            if self.classes[i] in self.filtro:
                pfilter.append(i)

        prop = []
        for p in pfilter:
            prop.append(result[p])

        return prop


    def getAttributes(self, texto):
        result = self.getCleanArray()
        
        tokens = texto.split(' ')
        for token in tokens:
            if token not in self.liwc:
                continue

            for prop in self.liwc[token]:
                idx = self.classes.index(prop)

                result[idx] = result[idx] + 1


        if len(self.filtro) > 0:
            return self.getFilter(result)

        return result


    def getCleanArray(self):
        newArray = []

        for i in self.classes:
            newArray.append(0)

        return newArray
    

def initLiwc(liwcPath, atributos = []):
    liwc = json.load(open(liwcPath))

    return Liwc(liwc, atributos)
