import numpy as np
import random

# Расширенный код Голея
b_matrix = np.array([
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])


# Формирование порождающей матрицы расширенного кода Голея
def G_matrix_goley(n, k, x):
    G = np.zeros((k, n), dtype=int)
    I_k = np.eye(k)
    for i in range(k):
        for j in range(k):
            G[i, j] = I_k[i, j]
        for m in range(n - k):
            G[i, k + m] = x[i, m]
    return G


# Формирование проверочной матрицы расширенного кода Голея
def H_matrix_goley(G):
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


# Получение синдрома ошибки
def get_syndroms(G, H):
    return G.dot(H) % 2


# Генерация слова с указанным количеством ошибок
def generate_word_with_n_mistakes(G, error_count):
    u = np.zeros(len(G), dtype=int)
    for i in range(len(u)):
        u[i] = random.randint(0, 1)
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


# Исправление одинарной ошибки
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


# Исправление двойной ошибки
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
                break
        if k >= 0:
            break
    if k == -1:
        print("Такого синдрома нет в матрице синдромов", '\n')
    else:
        slovo[k] += 1
        slovo[k] %= 2
        if d != -1:
            slovo[d] += 1
            slovo[d] %= 2
    return slovo


# Исправление тройной ошибки
def get_correct_word_three_mistakes(H, sindrom, slovo):
    k = -1
    d = -1
    g = -1
    for i in range(len(H)):
        if np.array_equal(sindrom, H[i]):
            k = i
            break
        for j in range(i + 1, len(H)):
            if np.array_equal(sindrom, H[i] + H[j]):
                k = i
                d = j
                break
            for e in range(j + 1, len(H)):
                if np.array_equal(sindrom, H[i] + H[j] + H[e]):
                    k = i
                    d = j
                    g = e
                    break
            if k >= 0:
                break
        if k >= 0:
            break
    if k == -1:
        print("Такого синдрома нет в матрице синдромов", '\n')
    else:
        slovo[k] += 1
        slovo[k] %= 2
        if d != -1:
            slovo[d] += 1
            slovo[d] %= 2
        if g != -1:
            slovo[g] += 1
            slovo[g] %= 2
    return slovo


# Исправление четверной ошибки
def get_correct_word_four_mistakes(H, sindrom, slovo):
    k = -1
    d = -1
    g = -1
    z = -1
    for i in range(len(H)):
        if np.array_equal(sindrom, H[i]):
            k = i
            break
        for j in range(i + 1, len(H)):
            if np.array_equal(sindrom, H[i] + H[j]):
                k = i
                d = j
                break
            for e in range(j + 1, len(H)):
                if np.array_equal(sindrom, H[i] + H[j] + H[e]):
                    k = i
                    d = j
                    g = e
                    break
                for y in range(e + 1, len(H)):
                    if np.array_equal(sindrom, H[i] + H[j] + H[e] + H[y]):
                        k = i
                        d = j
                        g = e
                        z = y
                        break
                if k >= 0:
                    break
            if k >= 0:
                break
        if k >= 0:
            break
    if k == -1:
        print("Такого синдрома нет в матрице синдромов", '\n')
    else:
        slovo[k] += 1
        slovo[k] %= 2
        if d != -1:
            slovo[d] += 1
            slovo[d] %= 2
        if g != -1:
            slovo[g] += 1
            slovo[g] %= 2
        if z != -1:
            slovo[z] += 1
            slovo[z] %= 2
    return slovo

# Порождающая матрица кода Рида-Маллера
def G_Rid_Maller(r, m):
    if 0 < r < m:
        G11_2 = G_Rid_Maller(r, m-1)
        G22 = G_Rid_Maller(r-1,m-1)
        G_left = np.concatenate([G11_2, np.zeros((len(G22), len(G11_2.T)), int)])
        G_right = np.concatenate([G11_2, G22])
        return np.concatenate([G_left, G_right], axis=1)
    elif r == 0:
        return np.ones((1, 2 ** m), dtype=int)
    elif r == m:
        G_top = G_Rid_Maller(m-1, m)
        bottom_matrix = np.zeros((1, 2 ** m), dtype=int)
        bottom_matrix[0][len(bottom_matrix) - 1] = 1
        return np.concatenate([G_top, ])

if __name__ == '__main__':
    G = G_matrix_goley(24, 12, b_matrix)
    print("Порождающая матрица G (24, 12, 8):", '\n', G, '\n')

    H = H_matrix_goley(G)
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

    word_with_three_mistakes = generate_word_with_n_mistakes(G, 3)
    print("Кодовое слово с тремя ошибками", '\n', word_with_three_mistakes, '\n')

    S = get_syndroms(word_with_three_mistakes, H)
    print("Синдром для кодового слова с тремя ошибками", '\n', S, '\n')

    correct_word_with_three_mistake = get_correct_word_three_mistakes(H, S, word_with_three_mistakes)
    print("Исправленное кодовое слово c тремя ошибками", '\n', correct_word_with_three_mistake, '\n')

    test = np.dot(correct_word_with_three_mistake, H) % 2
    print("Проверка (умножение исправленного слова на матрицу H)", '\n', test, '\n')

    word_with_four_mistakes = generate_word_with_n_mistakes(G, 4)
    print("Кодовое слово с четыремя ошибками", '\n', word_with_four_mistakes, '\n')

    S = get_syndroms(word_with_four_mistakes, H)
    print("Синдром для кодового слова с четырьмя ошибками", '\n', S, '\n')

    correct_word_with_four_mistake = get_correct_word_four_mistakes(H, S, word_with_four_mistakes)
    print("Исправленное кодовое слово c четырьмя ошибками", '\n', correct_word_with_four_mistake, '\n')

    test = np.dot(correct_word_with_four_mistake, H) % 2
    print("Проверка (умножение исправленного слова на матрицу H)", '\n', test, '\n')

    Rid_Maller = G_Rid_Maller(1, 3)
    print("RED MILLER", Rid_Maller)
