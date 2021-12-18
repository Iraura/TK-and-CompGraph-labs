import numpy as np
import random


def coding_with_errors(g, n, error_count):
    u = np.zeros(n, dtype=int)
    for i in range(n):
        u[i] = random.randint(0, 1)
    print("Исходное слово", u)
    result = np.polymul(u, g)
    print("polimul", result)
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


if __name__ == '__main__':
    print("lab6")
    g1 = np.array([1, 1, 0, 1])
    print("g1", g1)
    g2 = np.array([1, 0, 0, 1, 1, 1, 1])

    u = coding_with_errors(g1, 4, 2)
    print(u)
