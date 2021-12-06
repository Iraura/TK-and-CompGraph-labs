import numpy as np


# формируем бинарную матрицу размерности m столбцов
def get_basis_words(cols):
    matrix = []
    for i in range(2 ** cols):
        newRow = []
        for j in range(cols):
            newRow.append(i % 2)
            i = i // 2
        matrix.append(newRow)
    return np.asarray(matrix)


if __name__ == '__main__':
    a = get_basis_words(5)
    print(a)
