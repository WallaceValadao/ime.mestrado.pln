
def converterArray(array):
    minUser = 0

    for item in array:
        for atual in item:
            if atual < minUser:
                minUser = atual

    minUser = minUser * -1

    for item in array:
        for i in range(0, len(item)):
            item[i] += minUser

    return array