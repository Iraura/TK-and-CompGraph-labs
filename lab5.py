import numpy as np
from itertools import product
from itertools import combinations


# формируем бинарную матрицу размерности m столбцов
def get_basis(cols):
    return list(product([0, 1], repeat=cols))


# Векторная форма, формируем v
def get_V_I(I, m):
    if len(I) == 0:
        return np.ones(2 ** m, int)
    else:
        v = np.zeros(2 ** m, int)
        index = 0
        for words in get_basis(m):
            f = 1
            for j in I:
                f *= (words[j] + 1) % 2
            v[index] = f
            index += 1
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


if __name__ == '__main__':
    A = get_basis(3)
    print(A, sep='\n')

    print()
    print("v = ")
    I = get_I_combinations(4, 3)
    print(I)
    v = get_V_I(I[0], 3)
    print(v)
