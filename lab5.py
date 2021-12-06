import numpy as np
from itertools import product
from itertools import combinations
from operator import itemgetter
import math


# формируем бинарную матрицу размерности m столбцов
def get_basis(cols):
    return list(product([0, 1], repeat=cols))

def f(words, I):
    f = 1
    for j in I:
        f *= (words[j] + 1) % 2
    return f

# Векторная форма, формируем v
def get_V_I(I, m):
    if len(I) == 0:
        return np.ones(2 ** m, int)
    else:
        v = []
        for words in get_basis(m):
            f(words, I)
            v.append(f)
        return v


def get_I_combinations(m, r):
    multiplicity = np.zeros(m, int)
    boolean = list()
    for i in range(len(multiplicity)):
        multiplicity[i] = i

    for j in range(len(multiplicity) + 1):
        temp = list(combinations(multiplicity, j))
        for i in temp:
            if len(i) <= r:
                boolean.append(i)
    return boolean

def sort_I(I, m):
    r = 0
    result = []
    for i in range(len(I)):
        if len(I[i]) > r:
            r = len(I[i])

    for k in range(r + 1):
        s = 0
        g = m - 1
        for t in range(k):
            s += g
            g = g - 1
        for j in range(len(I)):
            for b in range(len(I)):
                if len(I[b]) == k:
                    sum = 0
                    for p in range(k):
                        sum += I[b][p]
                    if (sum == s):
                        if result.__contains__(I[b]):
                            continue
                        result.append(I[b])
                        if s != 0:
                            s = s - 1
    return result

def Rid_Maller_size(r, m):
    size = 0
    for i in range(r + 1):
        size += math.comb(m, i)
    return size


def Rid_Maller(r, m):
    matrix = np.zeros((Rid_Maller_size(r, m), 2 ** m), dtype=int)
    index = 0
    for i in get_I_combinations(m, r):
        matrix[index] = get_V_I(i, m)
        index += 1
    return matrix

def get_Komplement(I, m):
    komplement = []
    for i in range(m):
        if i not in I:
            komplement.append(i)
    return komplement

def get_H_I(I, m):
    H_I = []
    for words in get_basis(m):
        f = 1
        for i in I:
            f *= (words[i] + 1) % 2
        if f == 1:
            H_I.append(words)
    return H_I



if __name__ == '__main__':

    rm = Rid_Maller(3,4)
    print(rm)