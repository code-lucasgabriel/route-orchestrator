import math

def calculate_adjacency_matrix(V):
    n = len(V)
    A = [[0 for i in range(n)] for i in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i==j:
                A[i][j] = 0
            else:
                (x_i, y_i) = V[i]
                (x_j, y_j) = V[j]
                A[i][j] = math.sqrt((x_i-x_j)**2 + (y_i-y_j)**2)
    
    return A
