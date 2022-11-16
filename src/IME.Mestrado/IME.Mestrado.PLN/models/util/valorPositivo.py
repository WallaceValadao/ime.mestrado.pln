
def converterArray(array):
    minUser = 0

    for item in array:
        for item2 in item:
            for atual in item2:
                if atual < minUser:
                    minUser = atual

    minUser = minUser * -1

    for item in array:
        for item2 in item:
            for i in range(0, len(item2)):
                item2[i] += minUser

    return array