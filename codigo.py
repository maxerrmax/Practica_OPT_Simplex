import numpy as np

class Simplex:
    """
    Clase Simplex para definir la estructura principal del algoritmo
    del simplex primal. Recibe el vectores de coeficientes c, el vector
    de términos independientes b y la matriz A, donde c, A y b son arrays
    de NumPy.
    """

    def __init__(self, c, A, b):
        """
        Inicialización de la classe Simplex con los tres argumentos
        proporcionados por el enunciado.
        """

        self.c = c
        self.A = A
        self.b = b
    
    def fase_1(self, iteracion):
        """
        Implementación de la fase 1 del algoritmo del simplex primal. El objetivo
        es encontrar una SBF inicial con la que empezar en el problema original.
        """

        m = self.A.shape[0] 
        n = self.A.shape[1]

        # Inicializamos, a partir de los datos del enunciado, las matrices
        # y vectores para el problema artificial de la fase I.
        I = np.eye(m)
        A_f1 = np.hstack((self.A, I)) # Añadimos a la matriz A las columnas para las
                                      # varibles artificiales.

        c_f1 = np.concatenate((np.zeros(n), np.ones(m)))
        B_f1_idx = np.arange(n + 1, n + m + 1)
        N_f1_idx = np.arange(1, n + 1)
        C_B_f1 = c_f1[B_f1_idx - 1]
        B_f1 = np.eye(m)

        Xb_f1 = B_f1 @ self.b
        z_f1 = C_B_f1 @ Xb_f1

        # Resolvemos el problema artificial con el algoritmo del simplex primal y
        # utilizando como SBF inicial la formada por la variables artificiales. El algoritmo
        # está implementado en el método "fase_2".
        resultado = self.fase_2(B_f1_idx,N_f1_idx,Xb_f1,z_f1,B_f1,A_f1,c_f1,iteracion)

        # En caso de encontrar una DBF de descenso no acotada, indicamos que el
        # problema original no es factible.
        if resultado is None:
            print("Problema original no factible")
            return None
        
        X_b, z, B_idx, N_idx, B_inv, A, c, r, iteracion = resultado

        # En caso de que la z artificial no sea zero, indicamos que el problema
        # original no es factible.
        if abs(z) > 1e-10:
            print(f"Iteracion {iteracion}:\nvariables basicas = {B_idx},\nvalores = {X_b},\nr = {r},\nz = {z}")
            print()
            print("Problema original no factible, z mayor que 0")
            return None
        else:
            return X_b, z, B_idx, N_idx, B_inv, A, c, r, iteracion
    
    def fase_2(self, B_idx, N_idx, X_b, z, B_inv, A, c, iteracion):
        """
        Implementación de la fase 2 del algoritmo del simplex primal, con regla de 
        Bland y actualización de la inversa. El objetivo es encontrar la solución
        óptima final en relación a la función a optimizar y a las restricciones.
        """

        ### Paso 1:
        # Definimos los vectores de coeficientes de las variables básicas y no básicas y la matriz A
        # de las variables no básicas.
        C_N = c[N_idx - 1]
        C_B = c[B_idx - 1]
        A_N = A[:, N_idx -  1]

        ### Paso 2:
        # Calculamos los costes reducidos e inicializamos p y q a None para
        # la primera iteración.
        r = C_N - (C_B @ (B_inv @ A_N))
        p, q = None, None

        # Mientras haya algún coste reducido negativo, iteramos.
        while np.any(r < -1e-10):

            primera_negativa = np.where(r < -1e-10)[0][0]

            # Descubirmos la variable que entra (basándonos en la regla de Bland).
            q = N_idx[primera_negativa]

            ### Paso 3:
            # Calculamos la DBF de descenso.
            d_b = -B_inv @ A[:, q - 1]
            
            # En el caso que todas las componentes de d_b sean positivas o cero, indicamos
            # que tenemos una DBF de descenso no acotada, y por tanto no hay solución.
            if np.all(d_b >= -1e-10):
                print(f'DBF de descenso no acotada => (PL) no acotado, no hay solucion,\ndb = {d_b}')
                return None
            
            ### Paso 4:
            # Calculamos la longitud de paso máximo
            idx_neg = np.where(d_b < -1e-10)[0]
            theta = np.min(-X_b[idx_neg]/d_b[idx_neg])
            
            ratios = -X_b[idx_neg]/d_b[idx_neg]
            
            # Seleccionamos la variable de salida siguiendo la regla de Bland.
            min_ratio_mask = np.abs(ratios - theta) < 1e-10
            candidatos = idx_neg[min_ratio_mask]
            p = candidatos[np.argmin(B_idx[candidatos])]

            ### Paso 5:
            # Hacemos las actualizaciones necesarias a las diferentes matrices y vectores. 
            X_b = X_b + theta*d_b
            X_b[p] = theta
            z = z + theta * r[primera_negativa]

            print(f"Iteracion {iteracion}:\nsale = {B_idx[p]}, entra = {q},\nvariables basicas = {B_idx},\nvalores = {X_b},\nlongitud de paso = {theta},\nr = {r},\nz = {z}")
            print()
            
            B_idx[p], N_idx[primera_negativa] = q, B_idx[p]
            C_N = c[N_idx - 1]
            C_B = c[B_idx - 1]
            A_N = A[:, N_idx -  1]

            # Calculamos la nueva matriz inversa con el método de actulización de
            # la inversa y actualizamos los costes reducidos.
            B_inv = self._calculo_inv(B_inv,d_b,p)
            r = C_N - (C_B @ (B_inv @ A_N)) 
            iteracion += 1

        return X_b, z, B_idx, N_idx, B_inv, A, c, r, iteracion

    def _calculo_inv(self,B_inv,d_b,p):
        """
        Método auxiliar para implementar la actualización de la inversa.
        """

        m = d_b.shape[0]
        E = np.eye(m)

        # Construimos la matriz E:
        for i in range(m):
            if i == p:
                E[i][p] = (-1.0 / d_b[p])
            else:
                E[i][p] = (-d_b[i] / d_b[p])

        B_inv = E @ B_inv

        return B_inv
    
def leer_archivo(archivo):
    """
    Función que nos permite leer un archivo, en el cual estaran definidos
    los vectores c y b y la matriz A del problema a resolver.
    """

    with open(archivo, 'r') as f:
        contenido = f.read()
    
    partes = contenido.replace('c=', 'SPLIT').replace('A=', 'SPLIT').replace('b=', 'SPLIT').split('SPLIT')

    # Guardamos como arrays de NumPy los datos del problema.
    c = np.fromstring(partes[1], sep=' ')
    b = np.fromstring(partes[3], sep=' ')

    m = len(b)
    A = np.fromstring(partes[2], sep=' ').reshape(m, -1)

    return A, b, c

def solver(A,b,c):
    """
    Función que se encarga de resolver un problema definido por los vectores
    c y b y la matriz A.
    """

    print('Inicio del Simplex Primal con regla de Bland y actualizacion de la inversa:')
    iteracion = 1

    # Inicializamos la clase Simplex con los parámetros dados.
    s = Simplex(c, A, b)

    print('\nFase 1:')
    resultado = s.fase_1(iteracion)

    # En el caso que la fase 1 indique que el problema no es factible,
    # terminamos.
    if resultado is None:
        return
    
    X_b, z, B_idx, N_idx, B_inv, A_2, c_2, r, iteracion = resultado
    print(f'Solucion basica factible encontrada, iteracion {iteracion-1}')

    print('\nFase 2:')
    n = A.shape[1]

    # Reconstruimos los indices de las variables no basicas para la fase 2.
    N_idx_f2 = np.array([i for i in range(1, n + 1) if i not in B_idx])
    z = float(c[B_idx-1] @ X_b)
    resultado2 = s.fase_2(B_idx,N_idx_f2,X_b,z,B_inv,A,c,iteracion)

    # En el caso que la fase 2 indique que el problema no tiene solución,
    # termminamos.
    if resultado2 is None:
        return
    
    X_b, z, B_idx, N_idx, B_inv, A, c, r,iteracion = resultado2

    # Escribimos los resultados finales.
    print(f'Solucion optima encontrada, iteracion {iteracion-1}, z = {z}')
    print("\nSolucion optima:")
    print(f"z* = {z}")
    print(f"B_idx* = {B_idx}")
    print(f"X_b* = {X_b}")
    print(f"r* = {r}")

# Indicar aquí el nombre del archivo que se quiere leer.
archivo = "problema4_49.txt"

A, b, c = leer_archivo(archivo)
solver(A, b, c)