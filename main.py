import random

import numpy as np

x_matrix = np.array([[1, 1, 1],
                     [1, 1, 0],
                     [1, 0, 1],
                     [0, 1, 1]])


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
def generate_word_with_one_mistake(G):
    u = np.zeros(len(G), dtype=int)
    for i in range(len(u)):
        u[i] = random.randint(0, 2)
    u = u.dot(G)
    u %= 2
    mistake_pos = random.randint(0, 2)
    u[mistake_pos] += 1
    u[mistake_pos] %= 2
    return u


# Task 2.5
def get_correct_word(H, sindrom, slovo):
    k = 0
    for i in range(len(H)):
        if (np.array_equal(sindrom, H[i])):
            k = i
    slovo[k] += 1
    slovo[k] %= 2
    return slovo


if __name__ == '__main__':
    G = G_matrix(7, 4, x_matrix)
    print("Пождающая матрица G (7,4,3):", '\n', G, '\n')

    H = H_matrix(G)
    print("Проверочная матрица H", '\n', H, '\n')

    S = get_syndroms(G, H)
    print("Матрица синдромов S", '\n', S, '\n')

    word_with_one_mistake = generate_word_with_one_mistake(G)
    print("Кодовое слово с одной ошибкой", '\n', word_with_one_mistake, '\n')

    S = get_syndroms(word_with_one_mistake, H)
    print("синдром для кодового слова с ошибкой", '\n', S, '\n')

    correct_word = get_correct_word(H, S, word_with_one_mistake)
    print("исправленное кодовое слово", '\n', correct_word, '\n')

    test = np.dot(correct_word, H) % 2
    print("TEST", '\n', test, '\n')
