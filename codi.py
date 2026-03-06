import numpy as np

class Simplex:

    def __init__(self, c, A, b):
        self.c = c
        self.A = A
        self.b = b
    
    def fase_1():
        return 
    
    def fase_2(self, B_idx, N_idx, X_b, z, B_inv):
        C_N = self.c[N_idx - 1]
        C_B = self.c[B_idx - 1]
        A_N = self.A[:, N_idx -  1]
        r = C_N - (C_B @ (B_inv @ A_N))
        while np.any(r < 0):
            primera_negativa = np.argmax(r < 0)
            q = N_idx[primera_negativa]
            d_b = -B_inv @ A_N[:, primera_negativa]
            if np.all(d_b >= 0):
                print('DBF de descenso no acotada => (PL) no acotado, no hay solución')
                return   
            idx_neg = np.where(d_b < 0)
            theta = np.min(-X_b[idx_neg]/d_b[idx_neg])
            p = np.argmin(-X_b[idx_neg]/d_b[idx_neg])
            X_b = X_b + theta*d_b
            X_b[p] = theta
            z = z + theta * r[primera_negativa]
            B_idx[p], N_idx[primera_negativa] = q, B_idx[p]
            C_N = self.c[N_idx - 1]
            C_B = self.c[B_idx - 1]
            A_N = self.A[:, N_idx -  1]
            B_inv = self._calcul_inv(B_inv,d_b,p+1)
            r = C_N - (C_B @ (B_inv @ A_N)) 
        return X_b, z, B_idx, N_idx

    def _calcul_inv(self,B_inv,d_b,p):
        m = d_b.shape[0]
        E = np.eye(m)
        for i in range(m):
            if i == p:
                E[i][p-1] = (-1 / d_b[p-1])
            else:
                E[i][p-1] = (-d_b[i] / d_b[p-1])
        B_inv = E@B_inv
        return B_inv
    
"""B_idx = np.array([1,2])
N_idx = np.array([3,4])
c = np.array([1,2,3,4])
A = np.array([[1,2,3,4],[5,6,7,8]])
B_inv = np.array([[2,2],[1,2]])

test = Simplex(c, A, 0)
test.fase_2(B_idx, N_idx, 0, 0, B_inv)"""

"""test = np.array([1, 2, 9, 8])
print(np.any(test < 0))"""

"""a = np.array([1,2,-5,6,-90,2])
N_idx = np.array([2, 3, 5, 8, 9, 10])
X = [1 2 3 4 5 6 7 8 9 10]
p = np.argmax(a < 0)
var_que_entra = N_idx[np.argmax(a < 0)]
print(p)
print(var_que_entra)"""

# theta = 90
B_idx = np.array([5, 2, 3])
d_b = np.array([-2, -4, -12])
X_b = np.array([1, 2, 3])
b = np.where(d_b<0)
print(np.min(-X_b[b]/d_b[b]))
print(np.argmin(-X_b[b]/d_b[b]))

print(B_idx.shape[0])
