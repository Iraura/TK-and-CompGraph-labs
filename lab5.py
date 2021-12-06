import numpy as np
from itertools import product
from itertools import combinations
import math


# формируем бинарную матрицу размерности m столбцов
def get_basis(cols):
    return list(product([0, 1], repeat=cols))


def f_meth(words, I):
    f = 1
    for j in I:
        f *= (words[j] + 1) % 2
    return f


def f_meth_t(words, I, t):
    f = 1
    for j in I:
        f *= (words[j] + t[j] + 1) % 2
    return f


# Векторная форма, формируем v
def get_V_I(I, m):
    if len(I) == 0:
        return np.ones(2 ** m, int)
    else:
        v = []
        for words in get_basis(m):
            f = f_meth(words, I)
            v.append(f)
        return v


def get_V_I_t(I, m, t):
    if len(I) == 0:
        return np.ones(2 ** m, int)
    else:
        v = []
        for words in get_basis(m):
            f = f_meth_t(words, I, t)
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
                    if (sum == s) or (s != 1 and s % 2 == 1 and sum == s + 1):
                        if result.__contains__(I[b]):
                            continue
                        result.append(I[b])
                        if s != 0:
                            s = s - 1

    for i in range(len(I)):
        if result.__contains__(I[i]):
            continue
        sum = 0
        for k in range(len(I[i])):
            sum += I[i][k]

        for t in range(len(result)):
            if len(result[t]) == len(I[i]):
                sum2 = 0
                for b in range(len(result[t])):
                    sum2 += result[t][b]
                if (sum2 == sum):
                    if result.__contains__(I[i]):
                        continue
                    result.insert(t + 1, I[i])

    return result


def Rid_Maller_size(r, m):
    size = 0
    for i in range(r + 1):
        size += math.comb(m, i)
    return size


def Rid_Maller(r, m):
    matrix = np.zeros((Rid_Maller_size(r, m), 2 ** m), dtype=int)
    index = 0
    for i in get_I_combinations(m, r):  # for i in sort_I(get_I_combinations(m, r), m):
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
        f = f_meth(words, I)
        if f == 1:
            H_I.append(words)
    return H_I


def major_algorithm(w, r, m):
    i = r
    curr_w = w
    dead_edge = 2 ** (m - r - 1) - 1
    mi = []
    check = True

    while check:
        for J in get_I_combinations(m, i):
            edge = 2 ** (m - i - 1)
            zero = 0
            one = 0
            for t in get_H_I(J, m):
                c = np.dot(curr_w, get_V_I_t(get_Komplement(J, m), m, t)) % 2
                if c == 0:
                    zero += 1
                if c == 1:
                    one += 1
                if zero > dead_edge and one > dead_edge:
                    print("Необходима повторная отправка сообщения")
                    return
                if zero > edge:
                    mi.append(0)
                if one > edge:
                    mi.append(1)
                    curr_w = (curr_w + get_V_I(J, m)) % 2
                if i > 0:
                    if len(curr_w) < dead_edge:
                        for J in get_I_combinations(m, r + 1):
                            mi.append(0)
                            check = False
                    i -= 1
                else:
                    check = False
            mi.reverse()
            return mi


if __name__ == '__main__':
    rm = Rid_Maller(2, 4)
    print(rm)
