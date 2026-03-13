import numpy as np

class Simplex:

    def __init__(self, c, A, b):
        self.c = c
        self.A = A
        self.b = b
    
    def fase_1(self,iteracion):
        m = self.A.shape[0] 
        n = self.A.shape[1] 
        I = np.eye(m)
        A_f1 = np.hstack((self.A, I))
        c_f1 = np.concatenate((np.zeros(n), np.ones(m)))
        B_f1_idx = np.arange(n + 1, n + m + 1)
        N_f1_idx = np.arange(1, n + 1)
        C_B_f1 = c_f1[B_f1_idx - 1]
        B_f1 = np.eye(m)
        Xb_f1 = B_f1 @ self.b
        z_f1 = C_B_f1 @ Xb_f1
        resultado = self.fase_2(B_f1_idx,N_f1_idx,Xb_f1,z_f1,B_f1,A_f1,c_f1,iteracion)
        if resultado is None:
            print("Problema original no factible")
            return None
        X_b, z, B_idx, N_idx, B_inv, A, c, r,iteracion= resultado
        if abs(z) > 1e-10:
            print(f"Iteracion {iteracion}:\nvariables basicas = {B_idx},\nvalores = {X_b},\nr = {r},\nz = {z}")
            print()
            print("Problema original no factible, z mayor que 0")
            return None
        else:
            return X_b, z, B_idx, N_idx, B_inv, A, c, r,iteracion
    
    def fase_2(self, B_idx, N_idx, X_b, z, B_inv,A,c,iteracion):
        C_N = c[N_idx - 1]
        C_B = c[B_idx - 1]
        A_N = A[:, N_idx -  1]
        r = C_N - (C_B @ (B_inv @ A_N))
        p, q = None, None
        while np.any(r < -1e-10):
            primera_negativa = np.where(r < -1e-10)[0][0]
            q = N_idx[primera_negativa]
            d_b = -B_inv @ A[:, q - 1]
            
            if np.all(d_b >= -1e-10):
                print(f'DBF de descenso no acotada => (PL) no acotado, no hay solucion,\ndb = {d_b}')
                return None
            
            idx_neg = np.where(d_b < -1e-10)[0]
            theta = np.min(-X_b[idx_neg]/d_b[idx_neg])
            
            ratios = -X_b[idx_neg]/d_b[idx_neg]

            min_ratio_mask = np.abs(ratios - theta) < 1e-10
            candidates = idx_neg[min_ratio_mask]
            p = candidates[np.argmin(B_idx[candidates])]

            
            X_b = X_b + theta*d_b
            X_b[p] = theta
            z = z + theta * r[primera_negativa]

            print(f"Iteracion {iteracion}:\nsale = {B_idx[p]}, entra = {q},\nvariables basicas = {B_idx},\nvalores = {X_b},\nlongitud de paso = {theta},\nr = {r},\nz = {z}")
            print()
            
            B_idx[p], N_idx[primera_negativa] = q, B_idx[p]
            C_N = c[N_idx - 1]
            C_B = c[B_idx - 1]
            A_N = A[:, N_idx -  1]
            B_inv = self._calculo_inv(B_inv,d_b,p)
            r = C_N - (C_B @ (B_inv @ A_N)) 
            iteracion += 1
        return X_b, z, B_idx, N_idx, B_inv, A, c, r,iteracion

    def _calculo_inv(self,B_inv,d_b,p):
        m = d_b.shape[0]
        E = np.eye(m)
        for i in range(m):
            if i == p:
                E[i][p] = (-1.0 / d_b[p])
            else:
                E[i][p] = (-d_b[i] / d_b[p])
        B_inv = E @ B_inv
        return B_inv
    
def leer_archivo(archivo):
    with open(archivo, 'r') as f:
        contenido = f.read()
    
    partes = contenido.replace('c=', 'SPLIT').replace('A=', 'SPLIT').replace('b=', 'SPLIT').split('SPLIT')

    c = np.fromstring(partes[1], sep=' ')
    b = np.fromstring(partes[3], sep=' ')

    m = len(b)
    A = np.fromstring(partes[2], sep=' ').reshape(m, -1)

    return A,b,c

def solver(A,b,c):
    print('Inicio del Simplex Primal con regla de Bland y actualizacion de la inversa:')
    iteracion = 1
    s = Simplex(c,A,b)
    print('\nFase 1:')
    resultado = s.fase_1(iteracion)
    if resultado is None:
        return
    X_b, z, B_idx, N_idx, B_inv, A_2, c_2, r, iteracion = resultado
    print(f'Solucion basica factible encontrada, iteracion {iteracion-1}')
    print('\nFase 2:')
    n = A.shape[1]
    N_idx_f2 = np.array([i for i in range(1, n + 1) if i not in B_idx])
    z = float(c[B_idx-1] @ X_b)
    resultado2 = s.fase_2(B_idx,N_idx_f2,X_b,z,B_inv,A,c,iteracion)
    if resultado2 is None:
        return
    X_b, z, B_idx, N_idx, B_inv, A, c, r,iteracion = resultado2
    print(f'Solucion optima encontrada, iteracion {iteracion-1}, z = {z}')
    print("\nSolucion optima:")
    print(f"z* = {z}")
    print(f"B_idx* = {B_idx}")
    print(f"X_b* = {X_b}")
    print(f"r* = {r}")

A, b, c = leer_archivo("problema3_49.txt") # python .\codigo.py > resultat.txt
solver(A, b, c)