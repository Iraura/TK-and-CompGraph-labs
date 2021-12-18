import numpy as np
from itertools import product
from itertools import combinations
import math
import random
from operator import itemgetter


# Формируем бинарную матрицу размерности m столбцов
def get_basis(cols):
    return list(product([0, 1], repeat=cols))

# Считаем f_I
def f_meth(words, I):
    f = 1
    for j in I:
        f *= (words[j] + 1) % 2
    return f

# Считаем f_I_t
def f_meth_t(words, I, t):
    f = 1
    for j in I:
        f *= (words[j] + t[j] + 1) % 2
    return f


# Получаем вектор v_I
def get_V_I(I, m):
    if len(I) == 0:
        return np.ones(2 ** m, int)
    else:
        v = []
        for words in get_basis(m):
            f = f_meth(words, I)
            v.append(f)
        return v

# Получаем вектор v_I_t
def get_V_I_t(I, m, t):
    if len(I) == 0:
        return np.ones(2 ** m, int)
    else:
        v = []
        for words in get_basis(m):
            f = f_meth_t(words, I, t)
            v.append(f)
        return v

# Получаем комбинации I
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

# Сортировка
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

# Сортировка для декодирования
def sort_for_major(m, r):
    iterable = np.zeros(m, dtype=int)
    for i in range(m):
        iterable[i] = i

    temp = list(combinations(iterable, r))
    if len(temp[0]) != 0:
        temp.sort(key=itemgetter(len(temp[0]) - 1))

    result = list(np.zeros(math.comb(m, r), dtype=int))
    for i in range(len(temp)):
        result[i] = temp[i]
    return result

# Размер порождающей матрицы кода Рида Маллера
def Rid_Maller_size(r, m):
    size = 0
    for i in range(r + 1):
        size += math.comb(m, i)
    return size

# Формирование порождающей матрицы G
def Rid_Maller(r, m):
    size = Rid_Maller_size(r, m)
    matrix = np.zeros((size, pow(2, m)), dtype=int)
    index = 0
    for i in get_I_combinations(m, r):  # for i in sort_I(get_I_combinations(m, r), m):
        matrix[index] = get_V_I(i, m)
        index += 1
    return matrix

# Формирование комплиментарного множества
def get_Komplement(I, m):
    komplement = []
    for i in range(m):
        if i not in I:
            komplement.append(i)
    return komplement

# Ищем строки, где f_I = 1
def get_H_I(I, m):
    H_I = []
    for words in get_basis(m):
        f = f_meth(words, I)
        if f == 1:
            H_I.append(words)
    return H_I


# Мажоритарное декодирование
def major_algorithm(w, r, m, size):
    i = r
    w_r = w
    Mi = np.zeros(size, dtype=int)
    max_weight = pow(2, m - r - 1) - 1
    index = 0
    while True:
        for J in sort_for_major(m, i):
            max_zeros_and_ones_count = pow(2, m - i - 1)
            zeros_count, ones_count = 0, 0
            for t in get_H_I(J, m):
                komp = get_Komplement(J, m)
                V = get_V_I_t(komp, m, t)
                c = np.dot(w_r, V) % 2

                if c == 0:
                    zeros_count += 1
                elif c == 1:
                    ones_count += 1

            if zeros_count > max_weight and ones_count > max_weight:
                return
            if zeros_count > max_zeros_and_ones_count:
                Mi[index] = 0
                index += 1
            if ones_count > max_zeros_and_ones_count:
                Mi[index] = 1
                index += 1
                V = get_V_I(J, m)
                w_r = (w_r + V) % 2

        if i > 0:
            if len(w_r) < max_weight:
                for J in sort_for_major(m, r + 1):
                    Mi[index] = 0
                    index += 1
                break
            i -= 1
        else:
            break
    reversed(Mi)
    return Mi

# Функция генерации n-кратной ошибки
def generate_word_with_n_mistakes(G, r, m, error_count):
    u = np.zeros(Rid_Maller_size(r, m), dtype=int)
    for i in range(len(u)):
        u[i] = random.randint(0, 1)
    print("Исходное слово", u)
    print()
    u = u.dot(G)
    u %= 2
    err_arr = np.full(error_count, len(u) + 1, dtype=int)
    for k in range(error_count):
        mistake_pos = random.randint(0, len(u) - 1)
        while mistake_pos in err_arr:
            mistake_pos = random.randint(0, len(u) - 1)
        err_arr[k] = mistake_pos
        u[mistake_pos] += 1
        u[mistake_pos] %= 2
    return u


if __name__ == '__main__':
    # rm = Rid_Maller(2, 4)
    # print(rm)
    # m, r = 4, 2

    G = Rid_Maller(2, 4)
    print("Порождающая матрица G(2, 4): \n\n", G)
    print("_____________________________ \n")

    # Эксперимент для однократной ошибки
    Err = generate_word_with_n_mistakes(G, 2, 4, 1)
    print("Слово с однократной ошибкой: \n", Err)
    print()

    Decoded_word = major_algorithm(Err, 2, 4, len(G))
    if Decoded_word is None:
        print("Необходима повторная отправка сообщения!!!")
        print()
    else:
        print("Исправленное слово: \n", Decoded_word)
        print()
        V2 = Decoded_word.dot(G) % 2
        print("Результат умножения исправленного слова на матрицу G: \n", V2)
        print()


    # Эксперимент для двукратной ошибки
    print("_____________________________ \n")
    Err = generate_word_with_n_mistakes(G, 2, 4, 2)
    print("Слово с двукратной ошибкой: \n", Err)
    print()

    Decoded_word = major_algorithm(Err, 2, 4, len(G))
    if Decoded_word is None:
        print("Необходима повторная отправка сообщения!!!")
        print()
    else:
        print("Исправленное слово: \n", Decoded_word)
        print()
        V2 = Decoded_word.dot(G) % 2
        print("Результат умножения исправленного слова на матрицу G: \n", V2)
        print()




