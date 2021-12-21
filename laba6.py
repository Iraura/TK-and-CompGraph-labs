import numpy as np
import random


def coding_with_errors(g, n, error_count):
    u = np.zeros(n, dtype=int)
    for i in range(n):
        u[i] = random.randint(0, 1)
    print("Исходное слово", u)
    result = np.polymul(u, g)
    # print("polimul", result)
    result %= 2

    err_arr = np.zeros(error_count, dtype=int)
    for k in range(error_count):
        mistake_pos = random.randint(0, len(result) - 1)
        while mistake_pos in err_arr:
            mistake_pos = random.randint(0, len(result) - 1)
        err_arr[k] = mistake_pos
        result[mistake_pos] += 1
        result[mistake_pos] %= 2
    return result


def decoding(g, t, w):
    n = len(w)
    s = np.polydiv(w, g)[1]  # остаток
    for i in range(n):
        e_x = np.zeros(n, dtype=int)
        e_x[i] = 1
        mult = np.polymul(s, e_x)

        s_i = np.polydiv(mult, g)[1]
        # wt(s_i)
        if sum(s_i) <= t:
            e_i = np.zeros(n, dtype=int)
            e_i[i] = 1
            e_x = np.polymul(e_i, s_i)
            return np.polyadd(e_x, w)
    return None


if __name__ == '__main__':
    print("lab6")
    n = 7
    k = 3
    t = 1
    g1 = np.array([1, 1, 0, 1])
    print("g1", g1)
    print()

    for i in range(1, 4):
        u = coding_with_errors(g1, 4, i)
        print(i, "mistakes")
        print(u)
        decoded = decoding(g1, t, u)
        print("decoded hueta multidick", decoded)

        print()



    # _______________________________________________
    n2 = 15
    k2 = 9
    t2 = 3
    g2 = np.array([1, 0, 0, 1, 1, 1, 1])
