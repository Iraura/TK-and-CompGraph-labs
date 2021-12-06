import numpy as np
from itertools import product


# формируем бинарную матрицу размерности m столбцов
def get_basis(cols):
    return list(product([0, 1], repeat=cols))

# генерируем подмножество I
def get_I(m):
    return 0

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


if __name__ == '__main__':
    A = get_basis(3)
    print(A, sep='\n')

    print()
    print("v = ")
    I = [1, 4, 5]
    v = get_V_I(I, 3)
    print(v)


