import numpy as np

x_matrix = np.array([[1, 1, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1]])

# Task 2.1
def G_matrix(n, k, x):
    G = np.zeros((k, n), dtype = int)
    I_k = np.eye(k)
    for i in range (k):
        for j in range (k):
            G[i, j] = I_k[i, j]
        for m in range (n-k):
            G[i, k + m] = x[i, m]
    return G

# Task 2.2
def H_matrix(G):
    H = np.zeros((len(G.T), len(G.T) - len(G)), dtype = int)
    x_h = np.zeros((len(G), len(G.T) - len(G)), dtype = int)
    I_n_k = np.eye(len(G.T) - len(G))

    for i in range (len(G)):
        for j in range (len(G.T) - len(G)):
            x_h [i,j] = G[i, j + len(G)]

    for i in range (len(G)):
        for j in range (len(G.T) - len(G)):
            H[i,j] = x_h[i, j]

    for i in range (len(G.T) - len(G)):
        for j in range (len(G.T) - len(G)):
            H[i+ len(G), j] = I_n_k[i, j]
    return H

if __name__ == '__main__':
    print("Матрица G (7,4,3):", '\n', G_matrix(7, 4, x_matrix), '\n')
    G = G_matrix(7, 4, x_matrix)
    H = H_matrix(G)
    print ("Проверочная матрица H", '\n', H, '\n')

