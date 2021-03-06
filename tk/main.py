import random

import numpy as np

x_matrix_first = np.array([[1, 1, 1],
                           [1, 1, 0],
                           [1, 0, 1],
                           [0, 1, 1]])

x_matrix_second = np.array([[0, 1, 1, 1, 1, 0, 0],
                            [1, 1, 0, 1, 1, 1, 1],
                            [0, 0, 1, 1, 0, 1, 1],
                            [1, 0, 1, 0, 1, 1, 0]])


# Task 2.1
def G_matrix(n, k, x):
    G = np.zeros((k, n), dtype=int)
    I_k = np.eye(k)
    for i in range(k):
        for j in range(k):
            G[i, j] = I_k[i, j]
        for m in range(n - k):
            G[i, k + m] = x[i, m]
    return G


# Task 2.2
def H_matrix(G):
    H = np.zeros((len(G.T), len(G.T) - len(G)), dtype=int)
    x_h = np.zeros((len(G), len(G.T) - len(G)), dtype=int)
    I_n_k = np.eye(len(G.T) - len(G))

    for i in range(len(G)):
        for j in range(len(G.T) - len(G)):
            x_h[i, j] = G[i, j + len(G)]

    for i in range(len(G)):
        for j in range(len(G.T) - len(G)):
            H[i, j] = x_h[i, j]

    for i in range(len(G.T) - len(G)):
        for j in range(len(G.T) - len(G)):
            H[i + len(G), j] = I_n_k[i, j]
    return H


# Task 2.3
def get_syndroms(G, H):
    return G.dot(H) % 2


# Task 2.4
def generate_word_with_n_mistakes(G, error_count):
    u = np.zeros(len(G), dtype=int)
    for i in range(len(u)):
        u[i] = random.randint(0, 1)
    u = u.dot(G)
    u %= 2
    err_arr = np.zeros(error_count, dtype=int)
    for k in range(error_count):
        mistake_pos = random.randint(0, len(u) - 1)
        while mistake_pos in err_arr:
            mistake_pos = random.randint(0, len(u) - 1)
        err_arr[k] = mistake_pos
        u[mistake_pos] += 1
        u[mistake_pos] %= 2
    return u


def get_correct_word_one_mistake(H, sindrom, slovo):
    k = -1
    for i in range(len(H)):
        if np.array_equal(sindrom, H[i]):
            k = i
    if k == -1:
        print("Такого синдрома нет в матрице Н", '\n')
    else:
        slovo[k] += 1
        slovo[k] %= 2
    return slovo


def get_correct_word_two_mistakes(H, sindrom, slovo):
    k = -1
    d = -1
    for i in range(len(H)):
        if np.array_equal(sindrom, H[i]):
            k = i
            break
        for j in range(i + 1, len(H)):
            if np.array_equal(sindrom, H[i] + H[j]):
                k = i
                d = j
    if k == -1:
        print("Такого синдрома нет в матрице синдромов", '\n')
    else:
        slovo[k] += 1
        slovo[k] %= 2
        if d != -1:
            slovo[d] += 1
            slovo[d] %= 2
    return slovo


def first_part():
    print("    Часть 1", '\n', "_________________", '\n')

    G = G_matrix(7, 4, x_matrix_first)
    print("Пождающая матрица G (7,4,3):", '\n', G, '\n')

    H = H_matrix(G)
    print("Проверочная матрица H", '\n', H, '\n')

    word_with_one_mistake = generate_word_with_n_mistakes(G, 1)
    print("Кодовое слово с одной ошибкой", '\n', word_with_one_mistake, '\n')

    S = get_syndroms(word_with_one_mistake, H)
    print("Синдром для кодового слова с ошибкой", '\n', S, '\n')

    correct_word_with_one_mistake = get_correct_word_one_mistake(H, S, word_with_one_mistake)
    print("Исправленное кодовое слово c одной ошибкой", '\n', correct_word_with_one_mistake, '\n')

    test = np.dot(correct_word_with_one_mistake, H) % 2
    print("Проверка (умножение исправленного слова на матрицу H)", '\n', test, '\n')

    word_with_two_mistakes = generate_word_with_n_mistakes(G, 2)
    print("Кодовое слово с двумя ошибками", '\n', word_with_two_mistakes, '\n')

    S = get_syndroms(word_with_two_mistakes, H)
    print("Синдром для кодового слова с двумя ошибками", '\n', S, '\n')

    correct_word_with_two_mistake = get_correct_word_two_mistakes(H, S, word_with_two_mistakes)
    print("Исправленное кодовое слово c двумя ошибками", '\n', correct_word_with_two_mistake, '\n')

    test = np.dot(correct_word_with_two_mistake, H) % 2
    print("Проверка (умножение исправленного слова на матрицу H)", '\n', test, '\n')


def second_part():
    print("    Часть 2", '\n', "_________________", '\n')

    G = G_matrix(11, 4, x_matrix_second)
    print("Пождающая матрица G (11,4,5):", '\n', G, '\n')

    H = H_matrix(G)
    print("Проверочная матрица H", '\n', H, '\n')

    word_with_one_mistake = generate_word_with_n_mistakes(G, 1)
    print("Кодовое слово с одной ошибкой", '\n', word_with_one_mistake, '\n')

    S = get_syndroms(word_with_one_mistake, H)
    print("Синдром для кодового слова с ошибкой", '\n', S, '\n')

    correct_word = get_correct_word_one_mistake(H, S, word_with_one_mistake)
    print("Исправленное кодовое слово с одной ошибкой", '\n', correct_word, '\n')

    test = np.dot(correct_word, H) % 2
    print("Проверка (умножение исправленного слова на матрицу H)", '\n', test, '\n')

    word_with_two_mistakes = generate_word_with_n_mistakes(G, 2)
    print("Кодовое слово с двумя ошибками", '\n', word_with_two_mistakes, '\n')

    S = get_syndroms(word_with_two_mistakes, H)
    print("Синдром для кодового слова с двумя ошибками", '\n', S, '\n')

    correct_word_with_two_mistake = get_correct_word_two_mistakes(H, S, word_with_two_mistakes)
    print("Исправленное кодовое слово c двумя ошибками", '\n', correct_word_with_two_mistake, '\n')

    test = np.dot(correct_word_with_two_mistake, H) % 2
    print("Проверка (умножение исправленного слова на матрицу H)", '\n', test, '\n')

    word_with_three_mistakes = generate_word_with_n_mistakes(G, 3)
    print("Кодовое слово с тремя ошибками", '\n', word_with_three_mistakes, '\n')

    S = get_syndroms(word_with_three_mistakes, H)
    print("Синдром для кодового слова с тремя ошибками", '\n', S, '\n')

    correct_word_with_three_mistake = get_correct_word_two_mistakes(H, S, word_with_three_mistakes)
    print("Исправленное кодовое слово c тремя ошибками", '\n', correct_word_with_three_mistake, '\n')

    test = np.dot(correct_word_with_three_mistake, H) % 2
    print("Проверка (умножение исправленного слова на матрицу H)", '\n', test, '\n')


if __name__ == '__main__':
    first_part()
    second_part()
