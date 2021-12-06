import numpy as np
from itertools import product


# формируем бинарную матрицу размерности m столбцов
def get_basis(cols):
    return list(product([0, 1], repeat=cols))

if __name__ == '__main__':
    A = get_basis(3)
    print(*A, sep='\n')
