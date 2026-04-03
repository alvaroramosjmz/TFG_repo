# ==============================================================================
#
#  Fichero base para la implementación de nuevos algoritmos bioinspirados
#  de Feature Selection sobre el dataset EEG Essex.
#
#  Ejemplo:
#  python MyAlgorithms.py ABC 20 30 4 100
#
# ==============================================================================

import math

from numpy import zeros, sqrt, sum, where
import time
import sys

# ==============================================================================
#  LECTURA DE ARGUMENTOS
#  El flag opcional -m activa el modo medición (guarda tiempos y energía).
#  Sin -m el programa muestra resultados por consola y pinta la gráfica.
# ==============================================================================

dic_args = {
    "alg":              1,
    "agents":           2,
    "iterations":       3,
    "processes":        4,
    "desired_features": 5,
}
measure_mode = False
if sys.argv[1] == "-m":
    measure_mode = True
    dic_args = {
        "alg":              2,
        "agents":           3,
        "iterations":       4,
        "processes":        5,
        "desired_features": 6,
    }

# ==============================================================================
#  PARÁMETROS GLOBALES DE LA FUNCIÓN DE COSTE
#  alpha: peso del error de clasificación (1 - kappa)
#  beta:  peso de la fracción de features seleccionadas
#  alpha + beta = 1
# ==============================================================================

ALPHA = 0.95
BETA  = 1 - ALPHA

# ==============================================================================
#  FUNCIONES COMUNES — nivel de módulo
#
#  IMPORTANTE: estas funciones DEBEN estar fuera del bloque
#  "if __name__ == '__main__'" porque multiprocessing.Pool las necesita
#  importar en los procesos hijos. Si estuvieran dentro del main,
#  los procesos hijos no podrían acceder a ellas y el programa fallaría.
# ==============================================================================

# ------------------------------------------------------------------------------
#  k-NN implementado a mano (sin sklearn) para compatibilidad con pickle/Pool
# ------------------------------------------------------------------------------

def most_common(lst):
    """
    Devuelve el elemento más frecuente de una lista.
    Se usa como regla de decisión por mayoría de votos del k-NN.

    Parámetros:
    lst -- lista de etiquetas de los k vecinos más cercanos

    Retorna:
    La etiqueta más frecuente
    """
    return max(set(lst), key=lst.count)


def euclidean(point, data):
    """
    Calcula la distancia euclídea entre un punto y todas las filas de data.
    Versión vectorizada con NumPy: opera sobre toda la matriz de una vez.

    Parámetros:
    point -- vector 1D con las features del punto de test  [F_S]
    data  -- matriz 2D con las muestras de entrenamiento   [N_train x F_S]

    Retorna:
    Vector 1D con la distancia de point a cada fila de data  [N_train]
    """
    return sqrt(sum((point - data)**2, axis=1))


class KNeighborsClassifier:
    """
    Clasificador k-NN implementado a mano.

    Motivo: la versión de sklearn tiene problemas de serialización con
    multiprocessing en algunos entornos. Esta implementación es
    directamente compatible con Pool.starmap().

    Uso:
        neigh = KNeighborsClassifier(k=100)
        neigh.fit(X_train, y_train)
        predictions = neigh.predict(X_test)
    """

    def __init__(self, k=5, dist_metric=euclidean):
        self.k           = k
        self.dist_metric = dist_metric

    def fit(self, X_train, y_train):
        """
        Memoriza los datos de entrenamiento.
        En k-NN no hay entrenamiento real (lazy learning).
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predice la clase de cada muestra de X_test.

        Para cada muestra de test:
          1. Calcula distancias a todas las muestras de entrenamiento
          2. Ordena por distancia ascendente
          3. Toma las k etiquetas más cercanas
          4. Devuelve la etiqueta más frecuente (voto por mayoría)

        Retorna:
        Lista de etiquetas predichas, una por muestra de test
        """
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted  = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])
        return list(map(most_common, neighbors))

    def evaluate(self, X_test, y_test):
        """Calcula la accuracy directamente (no se usa en la pipeline principal)."""
        y_pred   = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy


# ------------------------------------------------------------------------------
#  Utilidades sobre agentes
# ------------------------------------------------------------------------------

def count_features(agent):
    """
    Cuenta el número de features seleccionadas por un agente
    (número de 1s en el vector binario).

    Parámetros:
    agent -- vector binario de longitud MAX_FEATURES

    Retorna:
    Número entero de features seleccionadas
    """
    import numpy as np
    ind = np.where(agent == 1)[0]
    return len(ind)


# ------------------------------------------------------------------------------
#  Función de coste — núcleo de la evaluación
# ------------------------------------------------------------------------------

def cost_func(agent, alpha, beta,
              num_samples_train, num_samples_test,
              train_x, train_y, test_x, test_y,
              num_char, max_features):
    """
    Evalúa la calidad de un agente (subconjunto de features).

    Proceso:
      1. Extrae las columnas del dataset correspondientes a las features
         seleccionadas por el agente (donde agent == 1)
      2. Entrena un k-NN con k=100 sobre los datos reducidos
      3. Calcula la accuracy en el conjunto de test
      4. Convierte la accuracy en índice kappa: κ = (C·acc - 1) / (C - 1)
      5. Devuelve:  f = alpha·(1 - κ) + beta·(F_S / F_T)

    Parámetros:
    agent             -- vector binario [max_features] con las features elegidas
    alpha             -- peso del error de clasificación (0.95)
    beta              -- peso de la fracción de features (0.05)
    num_samples_train -- número de muestras de entrenamiento (177)
    num_samples_test  -- número de muestras de test (177)
    train_x           -- matriz de entrenamiento [num_samples_train x max_features]
    train_y           -- etiquetas de entrenamiento [num_samples_train]
    test_x            -- matriz de test [num_samples_test x max_features]
    test_y            -- etiquetas de test [num_samples_test]
    num_char          -- número de clases (3)
    max_features      -- total de features del dataset (3600)

    Retorna:
    f ∈ [0, 1] — cuanto más bajo, mejor (el algoritmo minimiza este valor)
    """
    inputY = train_y
    testY  = test_y

    ind = where(agent == 1)[0]

    # Caso borde: agente sin ninguna feature → peor coste posible
    if len(ind) == 0:
        return 1

    # Construir submatrices reducidas con solo las features seleccionadas
    inputX = zeros((num_samples_train, len(ind)), dtype=float)
    testX  = zeros((num_samples_test,  len(ind)), dtype=float)
    aux    = [i for i in range(len(ind))]
    inputX[:, aux] = train_x[:, ind]
    testX[:, aux]  = test_x[:,  ind]

    # Clasificar con k-NN (k=100)
    neigh = KNeighborsClassifier(k=100)
    neigh.fit(inputX, inputY)
    prediction = neigh.predict(testX)

    # Calcular accuracy
    num_success = 0
    for i in range(num_samples_test):
        if prediction[i] == testY[i]:
            num_success += 1
    acc = num_success / num_samples_test

    # Convertir accuracy en índice kappa
    # κ = (C·acc - 1) / (C - 1)   con C = num_char = 3
    # Si acc = 1/3 (azar) → κ = 0
    # Si acc = 1   (perfecto) → κ = 1
    value = acc - 1 / (num_char - 1) * (1 - acc)
    error = 1 - value

    # Función objetivo multiobjetivo escalarizada
    return alpha * error + beta * len(ind) / max_features


# ==============================================================================
#
#  ABC — Función auxiliar a nivel de módulo
#
#  abc_generate_neighbor: genera la solución vecina v_i según la ecuación ABC.
#  Debe estar a nivel de módulo para ser serializable por multiprocessing.
#
#  Ecuación de actualización:
#      v_ij = x_ij + phi_ij * (x_ij - x_kj),   phi_ij in [-1, 1]
#
#  Binarización con función sigmoide:
#      S(v) = 1 / (1 + e^{-v})
#      nuevo_bit = 1 si rand() < S(v), si no 0
#
# ==============================================================================
 
def abc_generate_neighbor(x_i, x_k, rng_seed):
    """
    Genera la solución vecina v_i de la fuente x_i según la ecuación del ABC.
    Ejecutada en los procesos hijos de Pool.starmap().
 
    Parámetros:
    x_i      -- solución actual fuente i  [MAX_FEATURES]  binario
    x_k      -- solución fuente k≠i       [MAX_FEATURES]  binario
    rng_seed -- semilla para el RNG del proceso hijo
 
    Retorna:
    v_bin -- solución vecina binarizada  [MAX_FEATURES]  binario
    """
    import numpy as np
 
    rng = np.random.default_rng(rng_seed)
 
    # Copiar x_i como punto de partida
    v_i = x_i.copy().astype(float)
 
    # Elegir dimensión j aleatoria a perturbar
    j = rng.integers(0, len(x_i))
 
    # phi in [-1, 1] uniforme
    phi = rng.uniform(-1.0, 1.0)
 
    # Ecuación de actualización solo en dimensión j
    v_i[j] = x_i[j] + phi * (x_i[j] - x_k[j])
 
    # Binarizar con sigmoide: S(v) = 1 / (1 + e^{-v})
    v_bin    = x_i.copy()
    s_v      = 1.0 / (1.0 + math.exp(-v_i[j]))
    v_bin[j] = 1 if rng.random() < s_v else 0
 
    return v_bin


# ==============================================================================
#
#  BOA — Función auxiliar a nivel de módulo
#
#  boa_move_butterfly: calcula la nueva posición de una mariposa.
#  Debe estar a nivel de módulo para ser serializable por multiprocessing.
#
#  Ecuación búsqueda global:
#      x_i^{t+1} = x_i^t + (r^2 * g* - x_i^t) * f_i
#
#  Ecuación búsqueda local:
#      x_i^{t+1} = x_i^t + (r^2 * x_j^t - x_k^t) * f_i
#
#  Binarización con sigmoide: S(v) = 1/(1+e^{-v})
#
# ==============================================================================

def boa_move_butterfly(x_i, g_star, x_j, x_k, fragrance_i, p_switch, rng_seed):
    """
    Calcula la nueva posición de la mariposa i según el BOA.
    Ejecutada en los procesos hijos de Pool.starmap().

    Parámetros:
    x_i          -- posición actual mariposa i  [MAX_FEATURES]  binario
    g_star       -- mejor solución global       [MAX_FEATURES]  binario
    x_j, x_k     -- dos mariposas aleatorias j≠k≠i (para búsqueda local)
    fragrance_i  -- fragancia actual de la mariposa i  (escalar)
    p_switch     -- probabilidad de búsqueda global (0.8)
    rng_seed     -- semilla para el RNG del proceso hijo

    Retorna:
    x_new -- nueva posición binarizada  [MAX_FEATURES]  binario
    """
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    r   = rng.random()

    # Calcular nueva posición continua
    x_cont = x_i.astype(float)

    if r < p_switch:
        # Búsqueda global: moverse hacia g*
        # x_i^{t+1} = x_i^t + (r^2 * g* - x_i^t) * f_i
        x_cont = x_i + (r**2 * g_star - x_i) * fragrance_i
    else:
        # Búsqueda local: paseo aleatorio con x_j y x_k
        # x_i^{t+1} = x_i^t + (r^2 * x_j - x_k) * f_i
        x_cont = x_i + (r**2 * x_j - x_k) * fragrance_i

    # Binarizar dimensión a dimensión con función sigmoide
    x_new = np.zeros(len(x_i), dtype=int)
    for d in range(len(x_i)):
        s_v      = 1.0 / (1.0 + math.exp(-float(x_cont[d])))
        x_new[d] = 1 if rng.random() < s_v else 0

    return x_new

# ==============================================================================
#
#  MFO — Función auxiliar a nivel de módulo
#
#  mfo_move_moth: calcula la nueva posición de una polilla en espiral
#  alrededor de su llama asignada.
#  Debe estar a nivel de módulo para ser serializable por multiprocessing.
#
#  Ecuación de movimiento espiral:
#      D_i  = |F_j - M_i|
#      M_i^{l+1} = D_i * e^{bt} * cos(2*pi*t) + F_j
#
#  Binarización con sigmoide: S(v) = 1/(1+e^{-v})
#
# ==============================================================================

def mfo_move_moth(m_i, f_j, b, t, rng_seed):
    """
    Calcula la nueva posición de la polilla m_i orbitando su llama f_j.
    Ejecutada en los procesos hijos de Pool.starmap().

    Parámetros:
    m_i      -- posición actual de la polilla i  [MAX_FEATURES]  binario
    f_j      -- posición de la llama j asignada  [MAX_FEATURES]  binario
    b        -- constante de la espiral logarítmica (b=1)
    t        -- número aleatorio en [r, 1], r ∈ [-1,-2]
    rng_seed -- semilla para el RNG del proceso hijo

    Retorna:
    m_new -- nueva posición binarizada de la polilla  [MAX_FEATURES]  binario
    """
    import numpy as np

    rng = np.random.default_rng(rng_seed)

    # Distancia entre polilla y llama (componente a componente)
    # D_i = |F_j - M_i|
    d_i = np.abs(f_j.astype(float) - m_i.astype(float))

    # Ecuación de movimiento espiral logarítmica
    # M_i^{l+1} = D_i * e^{bt} * cos(2*pi*t) + F_j
    m_cont = d_i * math.exp(b * t) * math.cos(2 * math.pi * t) + f_j.astype(float)

    # Binarizar con función sigmoide: S(v) = 1/(1+e^{-v})
    m_new = np.zeros(len(m_i), dtype=int)
    for d in range(len(m_i)):
        s_v      = 1.0 / (1.0 + math.exp(-float(m_cont[d])))
        m_new[d] = 1 if rng.random() < s_v else 0

    return m_new

# ==============================================================================
#
#  HHO — Función auxiliar a nivel de módulo
#
#  hho_update_hawk: calcula la nueva posición de un halcón según el HHO.
#  Implementa las 6 fases del algoritmo.
#  Debe estar a nivel de módulo para ser serializable por multiprocessing.
#
# ==============================================================================

def hho_update_hawk(x_i, cost_i, x_rabbit, x_m, x_rand,
                    E, alpha, beta,
                    num_samples_train, num_samples_test,
                    train_x, train_y, test_x, test_y,
                    num_char, max_features, rng_seed):
    """
    Calcula la nueva posición del halcón x_i según la fase del HHO.
    Ejecutada en los procesos hijos de Pool.starmap().

    En las Fases 5 y 6 se evalúan los candidatos Y y Z llamando
    directamente a cost_func desde el proceso hijo.

    Parámetros:
    x_i, cost_i  -- posición y coste actual del halcón i
    x_rabbit     -- mejor solución global (la presa)
    x_m          -- posición media de todos los halcones
    x_rand       -- halcón aleatorio de la población (para Fase 1)
    E            -- energía de escape de la presa (escalar)
    alpha, beta  -- pesos de cost_func
    ... datos del dataset ...
    rng_seed     -- semilla para el RNG del proceso hijo

    Retorna:
    (x_new, cost_new) -- nueva posición binarizada y su coste
    """
    import numpy as np
    from scipy.special import gamma

    rng = np.random.default_rng(rng_seed)
    D   = len(x_i)

    # Función sigmoide vectorizada para binarizar
    def binarize(v_cont):
        s   = 1.0 / (1.0 + np.exp(-v_cont.astype(float)))
        return (rng.random(D) < s).astype(int)

    # Función de vuelo de Lévy (Heidari et al., 2019, Ec. levy)
    def levy(d):
        beta_levy = 1.5
        sigma = (gamma(1 + beta_levy) * np.sin(np.pi * beta_levy / 2) /
                 (gamma((1 + beta_levy) / 2) * beta_levy *
                  2 ** ((beta_levy - 1) / 2))) ** (1 / beta_levy)
        u = rng.random(d) * sigma
        v = rng.random(d)
        lf = 0.01 * u / (np.abs(v) ** (1 / beta_levy))
        return lf

    abs_E = abs(E)

    if abs_E >= 1:
        # ------------------------------------------------------------------
        #  FASE 1 — EXPLORACIÓN
        #  El halcón busca al conejo posicionándose en perchas elevadas
        # ------------------------------------------------------------------
        q  = rng.random()
        r1 = rng.random(); r2 = rng.random()
        r3 = rng.random(); r4 = rng.random()

        if q >= 0.5:
            # Sub-estrategia 1: posicionarse cerca de otro halcón aleatorio
            v_cont = (x_rand.astype(float)
                      - r1 * np.abs(x_rand.astype(float)
                                    - 2 * r2 * x_i.astype(float)))
        else:
            # Sub-estrategia 2: posición aleatoria en el territorio
            # LB=0, UB=1 → r3*(LB + r4*(UB-LB)) = r3*r4
            v_cont = (x_rabbit.astype(float) - x_m.astype(float)
                      - r3 * r4 * np.ones(D))

        x_new  = binarize(v_cont)
        c_new  = cost_func(x_new, alpha, beta,
                           num_samples_train, num_samples_test,
                           train_x, train_y, test_x, test_y,
                           num_char, max_features)

    else:
        # ------------------------------------------------------------------
        #  FASES 3-6 — EXPLOTACIÓN
        # ------------------------------------------------------------------
        r  = rng.random()
        r5 = rng.random()
        J  = 2 * (1 - r5)   # fuerza de salto del conejo

        delta_x = x_rabbit.astype(float) - x_i.astype(float)

        if r >= 0.5 and abs_E >= 0.5:
            # --------------------------------------------------------------
            #  FASE 3 — CERCO SUAVE
            #  X_i = ΔX - E*|J*X_rabbit - X_i|
            # --------------------------------------------------------------
            v_cont = (delta_x
                      - E * np.abs(J * x_rabbit.astype(float)
                                   - x_i.astype(float)))
            x_new = binarize(v_cont)
            c_new = cost_func(x_new, alpha, beta,
                              num_samples_train, num_samples_test,
                              train_x, train_y, test_x, test_y,
                              num_char, max_features)

        elif r >= 0.5 and abs_E < 0.5:
            # --------------------------------------------------------------
            #  FASE 4 — CERCO DURO
            #  X_i = X_rabbit - E*|ΔX|
            # --------------------------------------------------------------
            v_cont = (x_rabbit.astype(float)
                      - E * np.abs(delta_x))
            x_new = binarize(v_cont)
            c_new = cost_func(x_new, alpha, beta,
                              num_samples_train, num_samples_test,
                              train_x, train_y, test_x, test_y,
                              num_char, max_features)

        elif r < 0.5 and abs_E >= 0.5:
            # --------------------------------------------------------------
            #  FASE 5 — CERCO SUAVE CON BUCEOS RÁPIDOS (Lévy)
            #  Y = X_rabbit - E*|J*X_rabbit - X_i|
            #  Z = Y + S × LF(D)
            #  X_i = argmin(f(X_i), f(Y), f(Z))
            # --------------------------------------------------------------
            y_cont = (x_rabbit.astype(float)
                      - E * np.abs(J * x_rabbit.astype(float)
                                   - x_i.astype(float)))
            Y      = binarize(y_cont)
            c_Y    = cost_func(Y, alpha, beta,
                               num_samples_train, num_samples_test,
                               train_x, train_y, test_x, test_y,
                               num_char, max_features)

            S      = rng.random(D)
            z_cont = y_cont + S * levy(D)
            Z      = binarize(z_cont)
            c_Z    = cost_func(Z, alpha, beta,
                               num_samples_train, num_samples_test,
                               train_x, train_y, test_x, test_y,
                               num_char, max_features)

            # Selección progresiva: elegir la mejor de las tres
            if c_Y < cost_i:
                x_new, c_new = Y, c_Y
            elif c_Z < cost_i:
                x_new, c_new = Z, c_Z
            else:
                x_new, c_new = x_i.copy(), cost_i

        else:
            # --------------------------------------------------------------
            #  FASE 6 — CERCO DURO CON BUCEOS RÁPIDOS (Lévy)
            #  Y = X_rabbit - E*|J*X_rabbit - X_m|
            #  Z = Y + S × LF(D)
            #  X_i = argmin(f(X_i), f(Y), f(Z))
            # --------------------------------------------------------------
            y_cont = (x_rabbit.astype(float)
                      - E * np.abs(J * x_rabbit.astype(float)
                                   - x_m.astype(float)))
            Y      = binarize(y_cont)
            c_Y    = cost_func(Y, alpha, beta,
                               num_samples_train, num_samples_test,
                               train_x, train_y, test_x, test_y,
                               num_char, max_features)

            S      = rng.random(D)
            z_cont = y_cont + S * levy(D)
            Z      = binarize(z_cont)
            c_Z    = cost_func(Z, alpha, beta,
                               num_samples_train, num_samples_test,
                               train_x, train_y, test_x, test_y,
                               num_char, max_features)

            if c_Y < cost_i:
                x_new, c_new = Y, c_Y
            elif c_Z < cost_i:
                x_new, c_new = Z, c_Z
            else:
                x_new, c_new = x_i.copy(), cost_i

    return x_new, c_new


# ==============================================================================
#
#  ARO — Función auxiliar a nivel de módulo
#
#  aro_move_rabbit: genera la posición candidata v_i del conejo i.
#  Debe estar a nivel de módulo para ser serializable por multiprocessing.
#
#  Fase 2 — Forrajeo en desvío:
#      v_i = x_j + R*(x_i - x_j) + round(0.5*(0.05+r1))*n1
#      R = L * c,  L = (e - e^{((t-1)/I)²})*sin(2π*r2)
#
#  Fase 3 — Escondite aleatorio:
#      b_ir = x_i + H * g_r * x_i
#      v_i  = x_i + R*(r4*b_ir - x_i)
#      H = ((I-t+1)/I)*r4
#
# ==============================================================================

def aro_move_rabbit(x_i, x_j, A_t, t, num_it, rng_seed):
    """
    Genera la posición candidata v_i del conejo i según el ARO.
    Ejecutada en los procesos hijos de Pool.starmap().

    Parámetros:
    x_i      -- posición actual del conejo i  [D]  binario
    x_j      -- conejo aleatorio j≠i          [D]  binario  (para Fase 2)
    A_t      -- factor de energía de la iteración (escalar, compartido)
    t        -- iteración actual (1-indexed)
    num_it   -- número total de iteraciones I
    rng_seed -- semilla para el RNG del proceso hijo

    Retorna:
    v_bin -- posición candidata binarizada  [D]  binario
    """
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    D   = len(x_i)

    # Vector de mapeo c: activa aleatoriamente dimensiones (prob 0.5 por dim)
    c = rng.integers(0, 2, size=D)

    # Operador de carrera L (decreciente con las iteraciones)
    r2 = rng.random()
    L  = (math.e - math.exp(((t - 1) / num_it) ** 2)) * math.sin(2 * math.pi * r2)

    # Operador R = L * c  (vector)
    R = L * c

    if A_t > 1:
        # ----------------------------------------------------------------------
        #  FASE 2 — FORRAJEO EN DESVÍO (exploración)
        #  v_i = x_j + R*(x_i - x_j) + round(0.5*(0.05+r1))*n1
        # ----------------------------------------------------------------------
        r1 = rng.random()
        n1 = rng.standard_normal(D)          # ruido gaussiano

        v_cont = (x_j.astype(float)
                  + R * (x_i.astype(float) - x_j.astype(float))
                  + round(0.5 * (0.05 + r1)) * n1)

    else:
        # ----------------------------------------------------------------------
        #  FASE 3 — ESCONDITE ALEATORIO (explotación)
        #  b_ir = x_i + H * g_r * x_i
        #  v_i  = x_i + R*(r4*b_ir - x_i)
        # ----------------------------------------------------------------------
        r4 = rng.random()
        r5 = rng.random()

        # H: parámetro de escondite decreciente
        H = ((num_it - t + 1) / num_it) * r4

        # g_r: vector con 1 solo en la dimensión k aleatoria
        k   = int(math.ceil(r5 * D)) - 1    # índice 0-based
        k   = max(0, min(k, D - 1))         # clamp por seguridad
        g_r = np.zeros(D)
        g_r[k] = 1.0

        # Madriguera aleatoria alrededor de x_i
        b_ir = x_i.astype(float) + H * g_r * x_i.astype(float)

        v_cont = (x_i.astype(float)
                  + R * (r4 * b_ir - x_i.astype(float)))

    # Binarizar con función sigmoide: S(v) = 1/(1+e^{-v})
    s_v   = 1.0 / (1.0 + np.exp(-v_cont))
    v_bin = (rng.random(D) < s_v).astype(int)

    return v_bin


# ==============================================================================
#  PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================

if __name__ == "__main__":

    import random as rand
    import numpy as np
    import csv
    from matplotlib import pyplot as plt
    from multiprocessing.pool import Pool
    from codecarbon import EmissionsTracker

    # --------------------------------------------------------------------------
    #  GESTIÓN DE ERRORES Y SINTAXIS
    # --------------------------------------------------------------------------

    # Lista de algoritmos disponibles — añadir aquí cada nuevo algoritmo
    AVAILABLE_ALGORITHMS = ["ABC", "BOA", "MFO", "HHO", "ARO"]

    def print_syntax():
        """Muestra la sintaxis correcta de ejecución."""
        algs = "/".join(AVAILABLE_ALGORITHMS)
        print(f"Sintaxis: MyAlgorithms.py [-m] <{algs}> "
              f"<agentes> <iteraciones> <procesos> [<features_deseadas>]")

    def manage_error(msg):
        """Muestra un mensaje de error y termina el programa."""
        print(msg)
        print_syntax()
        sys.exit(2)

    # --------------------------------------------------------------------------
    #  LECTURA Y VALIDACIÓN DE ARGUMENTOS
    # --------------------------------------------------------------------------

    # Número de features objetivo (por defecto 100, configurable por argumento)
    DESIRED_N_FEATURES = 100

    if (len(sys.argv) == 7 and measure_mode) or (len(sys.argv) == 6 and not measure_mode):
        try:
            DESIRED_N_FEATURES = int(sys.argv[dic_args["desired_features"]])
        except ValueError:
            manage_error("Error. El número de features debe ser un entero.")

    try:
        num_proc = int(sys.argv[dic_args["processes"]])
    except ValueError:
        manage_error("Error. El número de procesos debe ser un entero.")
    except IndexError:
        manage_error("Error. No se ha proporcionado el número de procesos.")

    try:
        num_it = int(sys.argv[dic_args["iterations"]])
    except ValueError:
        manage_error("Error. El número de iteraciones debe ser un entero.")
    except IndexError:
        manage_error("Error. No se ha proporcionado el número de iteraciones.")

    try:
        num_ind = int(sys.argv[dic_args["agents"]])
    except ValueError:
        manage_error("Error. El número de agentes debe ser un entero.")
    except IndexError:
        manage_error("Error. No se ha proporcionado el número de agentes.")
    
    if num_ind < 2:
        manage_error("Error. N debe ser al menos 2 (SN = N/2 >= 1 fuente).")

    if DESIRED_N_FEATURES <= 0 or num_it <= 0 or num_ind <= 0:
        manage_error("Error. Todos los parámetros enteros deben ser mayores que 0.")

    alg_name = sys.argv[dic_args["alg"]].upper()
    if alg_name not in AVAILABLE_ALGORITHMS:
        manage_error(f"Error. Algoritmo '{alg_name}' no reconocido.")

    # --------------------------------------------------------------------------
    #  CONSTANTES DEL DATASET
    # --------------------------------------------------------------------------

    NUM_CHAR         = 3     # número de clases del dataset (1, 2, 3)
    MAX_FEATURES     = 3600  # número total de features
    NUM_SAMPLES_TRAIN = 178  # muestras de entrenamiento
    NUM_SAMPLES_TEST  = 178  # muestras de test

    # --------------------------------------------------------------------------
    #  CARGA DE DATOS
    # --------------------------------------------------------------------------

    TRAIN_X = np.empty((NUM_SAMPLES_TRAIN, MAX_FEATURES))
    TRAIN_Y = np.empty(NUM_SAMPLES_TRAIN)
    TEST_X  = np.empty((NUM_SAMPLES_TEST,  MAX_FEATURES))
    TEST_Y  = np.empty(NUM_SAMPLES_TEST)

    try:
        with open('Essex/104_training_data.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                for j in range(MAX_FEATURES):
                    TRAIN_X[i, j] = row[j]
    except (ValueError, FileNotFoundError):
        manage_error("Error leyendo los datos de entrenamiento. "
                     "Comprueba que existe la carpeta Essex/ con los CSV.")

    try:
        with open('Essex/104_training_class.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                TRAIN_Y[i] = row[0]
    except (ValueError, FileNotFoundError):
        manage_error("Error leyendo las clases de entrenamiento.")

    try:
        with open('Essex/104_testing_data.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                for j in range(MAX_FEATURES):
                    TEST_X[i, j] = row[j]
    except (ValueError, FileNotFoundError):
        manage_error("Error leyendo los datos de test.")

    try:
        with open('Essex/104_testing_class.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                TEST_Y[i] = row[0]
    except (ValueError, FileNotFoundError):
        manage_error("Error leyendo las clases de test.")

    # Confirmación de inicio
    print(f"Algoritmo:   {alg_name}")
    print(f"Agentes:     {num_ind}")
    print(f"Iteraciones: {num_it}")
    print(f"Procesos:    {num_proc}")
    print(f"Features:    {DESIRED_N_FEATURES}")

    # --------------------------------------------------------------------------
    #  FUNCIONES AUXILIARES DEL MAIN
    #  (aquí sí pueden estar dentro del main porque no se pasan a Pool)
    # --------------------------------------------------------------------------

    def construct_agent(num_features):
        """
        Crea un agente binario aleatorio con exactamente num_features unos.
        Garantiza que no se repite ninguna feature (sin reemplazo).

        Parámetros:
        num_features -- número de features que tendrá el agente

        Retorna:
        agent -- vector binario numpy de longitud MAX_FEATURES
        """
        agent     = np.zeros(MAX_FEATURES, int)
        remaining = np.array([i for i in range(MAX_FEATURES)])
        for i in range(num_features):
            new_feature       = rand.choice(remaining)
            agent[new_feature] = 1
            remaining         = np.where(np.logical_not(agent))[0]
        return agent

    def delete_features(agent):
        """
        Mecanismo de reparación: si el agente tiene más features de las
        permitidas, elimina las sobrantes aleatoriamente.
        Si tiene <= DESIRED_N_FEATURES, lo devuelve sin cambios.

        Parámetros:
        agent -- vector binario de longitud MAX_FEATURES

        Retorna:
        agent -- vector binario corregido (in-place)
        """
        selected_features = np.where(agent == 1)[0]
        num1 = len(selected_features)
        if num1 <= DESIRED_N_FEATURES:
            return agent
        ind        = np.random.choice(selected_features,
                                      size=num1 - DESIRED_N_FEATURES,
                                      replace=False)
        agent[ind] = 0
        return agent

    def cost_to_acc(cost, agent):
        """
        Transforma el valor de coste de vuelta a accuracy.
        Es la función inversa de cost_func.

        Matemáticamente:
          error = (cost - beta·F_S/F_T) / alpha
          kappa = 1 - error
          acc   = ((C-1)·kappa + 1) / C

        Parámetros:
        cost  -- valor devuelto por cost_func
        agent -- el agente evaluado (necesario para conocer F_S)

        Retorna:
        accuracy ∈ [0, 1]
        """
        selected_features = count_features(agent)
        error = (cost - BETA * selected_features / MAX_FEATURES) / ALPHA
        value = 1 - error
        return ((NUM_CHAR - 1) * value + 1) / NUM_CHAR

    # --------------------------------------------------------------------------
    #  FUNCIONES DE ESCRITURA DE RESULTADOS
    # --------------------------------------------------------------------------

    def write_time(elapsed, filename):
        """Añade el tiempo de ejecución al CSV de tiempos (modo medición)."""
        with open(filename, "a", newline="") as csvfile:
            csv.writer(csvfile).writerow([elapsed])

    def write_accuracy(accuracy, filename):
        """Añade la accuracy al CSV de accuracies (modo medición)."""
        with open(filename, "a", newline="") as csvfile:
            csv.writer(csvfile).writerow([accuracy])

    def write_solution(solution, filename):
        """Añade el vector solución al CSV de soluciones (modo medición)."""
        with open(filename, "a", newline="") as csvfile:
            csv.writer(csvfile).writerow(solution)

    def write_output(solution, accuracy):
        """
        Escribe el resultado final en output.csv.
        Sobreescribe el fichero en cada ejecución.
        Formato:
          accuracy, <valor>
          solution, <3600 valores 0/1>
        """
        with open("output.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["accuracy", accuracy])
            writer.writerow(["solution", *solution])


    # ==========================================================================
    #
    #  ABC — Artificial Bee Colony
    #
    # ==========================================================================
 
    if alg_name == "ABC":
 
        def abc():
            """
            ABC para Feature Selection binaria.
 
            Colonia: SN = N/2 fuentes, SN empleadas, SN observadoras.
 
            Por iteración:
              Fase 1 — Empleadas:    generar vecina + selección voraz
              Fase 2 — Observadoras: selección probabilística + voraz
              Fase 3 — Exploradoras: abandono de fuentes agotadas
              Fase 4 — Actualizar mejor global
 
            Retorna: (best_solution [MAX_FEATURES], curve [num_it])
            """
 
            # ------------------------------------------------------------------
            #  HIPERPARÁMETROS
            # ------------------------------------------------------------------
 
            SN    = num_ind // 2
            # limit: intentos fallidos máximos antes de abandonar una fuente
            # Valor estándar adaptado para FS: SN * DESIRED_N_FEATURES
            limit = SN * DESIRED_N_FEATURES
 
            # ------------------------------------------------------------------
            #  FASE 0 — INICIALIZACIÓN
            # ------------------------------------------------------------------
 
            X     = [construct_agent(DESIRED_N_FEATURES) for _ in range(SN)]
            costs = np.full(SN, np.inf)
            trial = np.zeros(SN, int)
 
            # Evaluación inicial en paralelo
            args = [[X[i], ALPHA, BETA,
                     NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST,
                     TRAIN_X, TRAIN_Y, TEST_X, TEST_Y,
                     NUM_CHAR, MAX_FEATURES] for i in range(SN)]
 
            with Pool(processes=num_proc) as pool:
                result = pool.starmap(cost_func, args,
                                      chunksize=SN // num_proc + int(SN % num_proc != 0))
            for i in range(SN):
                costs[i] = result[i]
 
            best_idx      = int(np.argmin(costs))
            best_solution = X[best_idx].copy()
            best_cost     = costs[best_idx]
            curve         = np.zeros(num_it)
 
            if not measure_mode:
                print(f"Iteración 0 | Mejor coste = {best_cost:.6f}")
 
            # ------------------------------------------------------------------
            #  BUCLE PRINCIPAL
            # ------------------------------------------------------------------
 
            for t in range(num_it):
 
                # --------------------------------------------------------------
                #  FASE 1 — ABEJAS EMPLEADAS
                # --------------------------------------------------------------
 
                # Generar vecinas en paralelo
                args_n = []
                for i in range(SN):
                    candidates = list(range(SN))
                    candidates.remove(i)
                    k    = rand.choice(candidates)
                    seed = np.random.randint(0, 2**31)
                    args_n.append([X[i], X[k], seed])
 
                with Pool(processes=num_proc) as pool:
                    v_emp = pool.starmap(abc_generate_neighbor, args_n,
                                         chunksize=SN // num_proc + int(SN % num_proc != 0))
 
                # Corregir y evaluar en paralelo
                args_c = []
                for i in range(SN):
                    v_emp[i] = delete_features(v_emp[i])
                    args_c.append([v_emp[i], ALPHA, BETA,
                                   NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST,
                                   TRAIN_X, TRAIN_Y, TEST_X, TEST_Y,
                                   NUM_CHAR, MAX_FEATURES])
 
                with Pool(processes=num_proc) as pool:
                    costs_emp = pool.starmap(cost_func, args_c,
                                             chunksize=SN // num_proc + int(SN % num_proc != 0))
 
                # Selección voraz — en serie
                for i in range(SN):
                    if costs_emp[i] < costs[i]:
                        X[i]     = v_emp[i]
                        costs[i] = costs_emp[i]
                        trial[i] = 0
                    else:
                        trial[i] += 1
 
                # --------------------------------------------------------------
                #  FASE 2 — ABEJAS OBSERVADORAS
                # --------------------------------------------------------------
 
                # fit_i = 1/(1+f_i): menor coste → mayor probabilidad
                fit   = 1.0 / (1.0 + costs)
                probs = fit / np.sum(fit)
 
                # Cada observadora elige fuente y genera vecina
                args_n_obs  = []
                src_obs     = []
                for _ in range(SN):
                    s = int(np.random.choice(SN, p=probs))
                    src_obs.append(s)
                    candidates = list(range(SN))
                    candidates.remove(s)
                    k    = rand.choice(candidates)
                    seed = np.random.randint(0, 2**31)
                    args_n_obs.append([X[s], X[k], seed])
 
                with Pool(processes=num_proc) as pool:
                    v_obs = pool.starmap(abc_generate_neighbor, args_n_obs,
                                         chunksize=SN // num_proc + int(SN % num_proc != 0))
 
                args_c_obs = []
                for i in range(SN):
                    v_obs[i] = delete_features(v_obs[i])
                    args_c_obs.append([v_obs[i], ALPHA, BETA,
                                       NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST,
                                       TRAIN_X, TRAIN_Y, TEST_X, TEST_Y,
                                       NUM_CHAR, MAX_FEATURES])
 
                with Pool(processes=num_proc) as pool:
                    costs_obs = pool.starmap(cost_func, args_c_obs,
                                             chunksize=SN // num_proc + int(SN % num_proc != 0))
 
                for i in range(SN):
                    s = src_obs[i]
                    if costs_obs[i] < costs[s]:
                        X[s]     = v_obs[i]
                        costs[s] = costs_obs[i]
                        trial[s] = 0
                    else:
                        trial[s] += 1
 
                # --------------------------------------------------------------
                #  FASE 3 — ABEJAS EXPLORADORAS
                # --------------------------------------------------------------
 
                for i in range(SN):
                    if trial[i] > limit:
                        X[i]     = construct_agent(DESIRED_N_FEATURES)
                        costs[i] = cost_func(X[i], ALPHA, BETA,
                                             NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST,
                                             TRAIN_X, TRAIN_Y, TEST_X, TEST_Y,
                                             NUM_CHAR, MAX_FEATURES)
                        trial[i] = 0
 
                # --------------------------------------------------------------
                #  FASE 4 — ACTUALIZAR MEJOR GLOBAL
                # --------------------------------------------------------------
 
                idx = int(np.argmin(costs))
                if costs[idx] < best_cost:
                    best_solution = X[idx].copy()
                    best_cost     = costs[idx]
 
                curve[t] = best_cost
 
                if not measure_mode:
                    print(f"Iteración {t+1} | Mejor coste = {best_cost:.6f}")
 
            return best_solution, curve
        
    
    # ==========================================================================
    #
    #  BOA — Butterfly Optimization Algorithm
    #
    # ==========================================================================

    elif alg_name == "BOA":

        def boa():
            """
            BOA para Feature Selection binaria.

            Cada mariposa emite una fragancia proporcional a su fitness.
            En cada iteración se mueve hacia g* (global) o hace un paseo
            aleatorio con dos mariposas vecinas (local), con probabilidad p.

            Hiperparámetros (Arora & Singh, 2019):
              c     = 0.01   modalidad sensorial
              a_ini = 0.1    exponente inicial (exploración)
              a_fin = 0.3    exponente final   (explotación)
              p     = 0.8    probabilidad de búsqueda global

            Retorna: (g_star [MAX_FEATURES], curve [num_it])
            """

            # ------------------------------------------------------------------
            #  HIPERPARÁMETROS
            # ------------------------------------------------------------------

            C     = 0.01   # modalidad sensorial
            A_INI = 0.1    # exponente de potencia inicial
            A_FIN = 0.3    # exponente de potencia final
            P     = 0.8    # probabilidad de búsqueda global

            # ------------------------------------------------------------------
            #  FASE 0 — INICIALIZACIÓN
            # ------------------------------------------------------------------

            X      = [construct_agent(DESIRED_N_FEATURES) for _ in range(num_ind)]
            costs  = np.full(num_ind, np.inf)
            curve  = np.zeros(num_it)

            # Evaluación inicial en paralelo
            args = [[X[i], ALPHA, BETA,
                     NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST,
                     TRAIN_X, TRAIN_Y, TEST_X, TEST_Y,
                     NUM_CHAR, MAX_FEATURES] for i in range(num_ind)]

            with Pool(processes=num_proc) as pool:
                result = pool.starmap(cost_func, args,
                                      chunksize=num_ind // num_proc + int(num_ind % num_proc != 0))
            for i in range(num_ind):
                costs[i] = result[i]

            # Calcular fragancia inicial: I_i = 1/(1+f_i), frag_i = C * I_i^A_INI
            intensity  = 1.0 / (1.0 + costs)
            fragrances = C * (intensity ** A_INI)

            # Mejor solución inicial g*
            best_idx = int(np.argmin(costs))
            g_star   = X[best_idx].copy()
            best_cost = costs[best_idx]

            if not measure_mode:
                print(f"Iteración 0 | Mejor coste = {best_cost:.6f}")

            # ------------------------------------------------------------------
            #  BUCLE PRINCIPAL
            # ------------------------------------------------------------------

            for t in range(num_it):

                # Actualizar exponente a progresivamente (exploración → explotación)
                a = A_INI + (A_FIN - A_INI) * (t + 1) / num_it

                # --------------------------------------------------------------
                #  MOVIMIENTO DE CADA MARIPOSA (#parallel N, λ)
                # --------------------------------------------------------------

                args_move = []
                for i in range(num_ind):
                    # Para búsqueda local: elegir j,k aleatorios con j≠k≠i
                    candidates = list(range(num_ind))
                    candidates.remove(i)
                    j, k = rand.sample(candidates, 2)
                    seed = np.random.randint(0, 2**31)
                    args_move.append([X[i], g_star, X[j], X[k],
                                      fragrances[i], P, seed])

                with Pool(processes=num_proc) as pool:
                    X_new = pool.starmap(
                        boa_move_butterfly, args_move,
                        chunksize=num_ind // num_proc + int(num_ind % num_proc != 0)
                    )

                # Corregir features sobrantes
                for i in range(num_ind):
                    X_new[i] = delete_features(X_new[i])

                # --------------------------------------------------------------
                #  EVALUACIÓN EN PARALELO
                # --------------------------------------------------------------

                args_cost = [[X_new[i], ALPHA, BETA,
                              NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST,
                              TRAIN_X, TRAIN_Y, TEST_X, TEST_Y,
                              NUM_CHAR, MAX_FEATURES] for i in range(num_ind)]

                with Pool(processes=num_proc) as pool:
                    costs_new = pool.starmap(
                        cost_func, args_cost,
                        chunksize=num_ind // num_proc + int(num_ind % num_proc != 0)
                    )

                # Actualizar posiciones y fragancias (todas las mariposas se mueven)
                for i in range(num_ind):
                    X[i]     = X_new[i]
                    costs[i] = costs_new[i]

                intensity  = 1.0 / (1.0 + costs)
                fragrances = C * (intensity ** a)

                # --------------------------------------------------------------
                #  ACTUALIZAR MEJOR SOLUCIÓN GLOBAL g*
                # --------------------------------------------------------------

                current_best_idx = int(np.argmin(costs))
                if costs[current_best_idx] < best_cost:
                    g_star    = X[current_best_idx].copy()
                    best_cost = costs[current_best_idx]

                curve[t] = best_cost

                if not measure_mode:
                    print(f"Iteración {t+1} | Mejor coste = {best_cost:.6f}")

            return g_star, curve
        
    # ==========================================================================
    #
    #  MFO — Moth-Flame Optimization
    #
    # ==========================================================================

    elif alg_name == "MFO":

        def mfo():
            """
            MFO para Feature Selection binaria.

            Dos estructuras separadas:
              M -- polillas: agentes activos que se mueven en espiral
              F -- llamas:   mejores posiciones históricas (nunca empeoran)

            Mecanismos clave:
              - flame_no decrece de N a 1 (exploración → explotación)
              - r decrece de -1 a -2 (t más cercano a -1 → más explotación)
              - Llamas se actualizan combinando M y F y ordenando

            Hiperparámetros (Mirjalili, 2015):
              b = 1  (constante de la espiral logarítmica)

            Retorna: (F_1 [MAX_FEATURES], curve [num_it])
            """

            # ------------------------------------------------------------------
            #  HIPERPARÁMETROS
            # ------------------------------------------------------------------

            B = 1   # constante de la espiral logarítmica

            # ------------------------------------------------------------------
            #  FASE 0 — INICIALIZACIÓN
            # ------------------------------------------------------------------

            # Inicializar N polillas
            M      = [construct_agent(DESIRED_N_FEATURES) for _ in range(num_ind)]
            OM     = np.full(num_ind, np.inf)   # costes de las polillas
            curve  = np.zeros(num_it)

            # Evaluación inicial en paralelo
            args = [[M[i], ALPHA, BETA,
                     NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST,
                     TRAIN_X, TRAIN_Y, TEST_X, TEST_Y,
                     NUM_CHAR, MAX_FEATURES] for i in range(num_ind)]

            with Pool(processes=num_proc) as pool:
                result = pool.starmap(cost_func, args,
                                      chunksize=num_ind // num_proc + int(num_ind % num_proc != 0))
            for i in range(num_ind):
                OM[i] = result[i]

            # Ordenar polillas por fitness ascendente
            sorted_idx = np.argsort(OM)
            M  = [M[i] for i in sorted_idx]
            OM = OM[sorted_idx]

            # Llamas iniciales = polillas ordenadas
            F  = [M[i].copy() for i in range(num_ind)]
            OF = OM.copy()

            # Mejor solución: primera llama (la de menor coste)
            best_solution = F[0].copy()
            best_cost     = OF[0]

            if not measure_mode:
                print(f"Iteración 0 | Mejor coste = {best_cost:.6f}")

            # ------------------------------------------------------------------
            #  BUCLE PRINCIPAL
            # ------------------------------------------------------------------

            for l in range(1, num_it + 1):

                # Número adaptativo de llamas activas (decrece de N a 1)
                flame_no = round(num_ind - l * (num_ind - 1) / num_it)
                flame_no = max(1, flame_no)   # mínimo 1 llama siempre

                # Constante de convergencia r ∈ [-1, -2]
                r = -1 - l / num_it

                # --------------------------------------------------------------
                #  MOVIMIENTO EN ESPIRAL (#parallel N, λ)
                # --------------------------------------------------------------

                args_move = []
                for i in range(num_ind):
                    # Asignación de llama: polilla i → llama i si i < flame_no,
                    # si no → última llama activa
                    j = i if i < flame_no else flame_no - 1

                    # t ~ U(r, 1): posición sobre la espiral
                    t = rand.uniform(r, 1)

                    seed = np.random.randint(0, 2**31)
                    args_move.append([M[i], F[j], B, t, seed])

                with Pool(processes=num_proc) as pool:
                    M_new = pool.starmap(
                        mfo_move_moth, args_move,
                        chunksize=num_ind // num_proc + int(num_ind % num_proc != 0)
                    )

                # Corregir features sobrantes
                for i in range(num_ind):
                    M_new[i] = delete_features(M_new[i])

                # --------------------------------------------------------------
                #  EVALUACIÓN EN PARALELO
                # --------------------------------------------------------------

                args_cost = [[M_new[i], ALPHA, BETA,
                              NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST,
                              TRAIN_X, TRAIN_Y, TEST_X, TEST_Y,
                              NUM_CHAR, MAX_FEATURES] for i in range(num_ind)]

                with Pool(processes=num_proc) as pool:
                    OM_new = pool.starmap(
                        cost_func, args_cost,
                        chunksize=num_ind // num_proc + int(num_ind % num_proc != 0)
                    )

                OM_new = np.array(OM_new)

                # Actualizar posiciones de polillas
                for i in range(num_ind):
                    M[i]  = M_new[i]
                    OM[i] = OM_new[i]

                # --------------------------------------------------------------
                #  ACTUALIZACIÓN DE LLAMAS
                #  Combinar polillas nuevas con llamas anteriores,
                #  ordenar y quedarse con las N mejores
                # --------------------------------------------------------------

                # Combinar las 2N soluciones (polillas + llamas anteriores)
                combined_agents = M + F
                combined_costs  = np.concatenate([OM_new, OF])

                # Ordenar por coste ascendente y tomar las N mejores
                sorted_idx = np.argsort(combined_costs)
                F  = [combined_agents[idx].copy() for idx in sorted_idx[:num_ind]]
                OF = combined_costs[sorted_idx[:num_ind]]

                # --------------------------------------------------------------
                #  ACTUALIZAR MEJOR SOLUCIÓN GLOBAL
                #  La mejor llama F[0] es siempre la mejor solución global
                # --------------------------------------------------------------

                if OF[0] < best_cost:
                    best_solution = F[0].copy()
                    best_cost     = OF[0]

                curve[l - 1] = best_cost

                if not measure_mode:
                    print(f"Iteración {l} | Mejor coste = {best_cost:.6f}")

            return best_solution, curve
        
    # ==========================================================================
    #
    #  HHO — Harris Hawks Optimization
    #
    # ==========================================================================

    elif alg_name == "HHO":

        def hho():
            """
            HHO para Feature Selection binaria.

            La presa X_rabbit es la mejor solución global.
            Cada halcón elige dinámicamente entre 6 fases según |E| y r:
              |E|>=1           → Fase 1: Exploración
              |E|<1, r>=0.5, |E|>=0.5 → Fase 3: Cerco suave
              |E|<1, r>=0.5, |E|<0.5  → Fase 4: Cerco duro
              |E|<1, r<0.5,  |E|>=0.5 → Fase 5: Cerco suave + Lévy
              |E|<1, r<0.5,  |E|<0.5  → Fase 6: Cerco duro + Lévy

            Hiperparámetros (Heidari et al., 2019):
              β = 1.5  (exponente del vuelo de Lévy, fijo en la función)

            Retorna: (X_rabbit [MAX_FEATURES], curve [num_it])
            """

            # ------------------------------------------------------------------
            #  FASE 0 — INICIALIZACIÓN
            # ------------------------------------------------------------------

            X      = [construct_agent(DESIRED_N_FEATURES) for _ in range(num_ind)]
            costs  = np.full(num_ind, np.inf)
            curve  = np.zeros(num_it)

            # Evaluación inicial en paralelo
            args = [[X[i], ALPHA, BETA,
                     NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST,
                     TRAIN_X, TRAIN_Y, TEST_X, TEST_Y,
                     NUM_CHAR, MAX_FEATURES] for i in range(num_ind)]

            with Pool(processes=num_proc) as pool:
                result = pool.starmap(cost_func, args,
                                      chunksize=num_ind // num_proc + int(num_ind % num_proc != 0))
            for i in range(num_ind):
                costs[i] = result[i]

            # Designar la presa como el mejor halcón
            best_idx  = int(np.argmin(costs))
            x_rabbit  = X[best_idx].copy()
            best_cost = costs[best_idx]

            if not measure_mode:
                print(f"Iteración 0 | Mejor coste = {best_cost:.6f}")

            # ------------------------------------------------------------------
            #  BUCLE PRINCIPAL
            # ------------------------------------------------------------------

            for t in range(1, num_it + 1):

                # Posición media del grupo (necesaria para Fases 1 y 6)
                X_matrix = np.array([X[i].astype(float) for i in range(num_ind)])
                x_m      = np.mean(X_matrix, axis=0)

                # Construir argumentos para cada halcón
                args_hawk = []
                for i in range(num_ind):
                    # E0 ~ U(-1,1), E = 2*E0*(1 - t/I)
                    E0 = rand.uniform(-1, 1)
                    E  = 2 * E0 * (1 - t / num_it)

                    # Halcón aleatorio distinto de i (para Fase 1)
                    candidates = list(range(num_ind))
                    candidates.remove(i)
                    x_rand = X[rand.choice(candidates)]

                    seed = np.random.randint(0, 2**31)
                    args_hawk.append([X[i], costs[i], x_rabbit, x_m, x_rand,
                                      E, ALPHA, BETA,
                                      NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST,
                                      TRAIN_X, TRAIN_Y, TEST_X, TEST_Y,
                                      NUM_CHAR, MAX_FEATURES, seed])

                # Actualizar halcones en paralelo
                # Nota: las Fases 5 y 6 llaman a cost_func internamente,
                # por lo que cada proceso puede hacer hasta 3 evaluaciones
                with Pool(processes=num_proc) as pool:
                    results_hawk = pool.starmap(
                        hho_update_hawk, args_hawk,
                        chunksize=num_ind // num_proc + int(num_ind % num_proc != 0)
                    )

                # Actualizar posiciones y costes
                for i in range(num_ind):
                    x_new, c_new = results_hawk[i]
                    x_new        = delete_features(x_new)
                    X[i]         = x_new
                    costs[i]     = c_new

                # Actualizar la presa X_rabbit
                current_best_idx = int(np.argmin(costs))
                if costs[current_best_idx] < best_cost:
                    x_rabbit  = X[current_best_idx].copy()
                    best_cost = costs[current_best_idx]

                curve[t - 1] = best_cost

                if not measure_mode:
                    print(f"Iteración {t} | Mejor coste = {best_cost:.6f}")

            return x_rabbit, curve


    # ==========================================================================
    #
    #  ARO — Artificial Rabbits Optimization
    #
    # ==========================================================================

    elif alg_name == "ARO":

        def aro():
            """
            ARO para Feature Selection binaria.

            Dos estrategias controladas por el factor de energía A(t):
              A(t) > 1 → Fase 2: Forrajeo en desvío (exploración)
              A(t) ≤ 1 → Fase 3: Escondite aleatorio (explotación)

            A(t) se calcula una sola vez por iteración y se aplica
            homogéneamente a todos los conejos.
            Selección greedy: cada conejo actualiza su posición solo si mejora.

            Sin hiperparámetros adicionales: A(t) controla la transición
            de forma automática.

            Retorna: (x_best [MAX_FEATURES], curve [num_it])
            """

            # ------------------------------------------------------------------
            #  FASE 0 — INICIALIZACIÓN
            # ------------------------------------------------------------------

            X      = [construct_agent(DESIRED_N_FEATURES) for _ in range(num_ind)]
            costs  = np.full(num_ind, np.inf)
            curve  = np.zeros(num_it)

            # Evaluación inicial en paralelo
            args = [[X[i], ALPHA, BETA,
                     NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST,
                     TRAIN_X, TRAIN_Y, TEST_X, TEST_Y,
                     NUM_CHAR, MAX_FEATURES] for i in range(num_ind)]

            with Pool(processes=num_proc) as pool:
                result = pool.starmap(cost_func, args,
                                      chunksize=num_ind // num_proc + int(num_ind % num_proc != 0))
            for i in range(num_ind):
                costs[i] = result[i]

            best_idx      = int(np.argmin(costs))
            best_solution = X[best_idx].copy()
            best_cost     = costs[best_idx]

            if not measure_mode:
                print(f"Iteración 0 | Mejor coste = {best_cost:.6f}")

            # ------------------------------------------------------------------
            #  BUCLE PRINCIPAL
            # ------------------------------------------------------------------

            for t in range(1, num_it + 1):

                # Factor de energía: único por iteración
                # A(t) = 4*(1 - t/I)*ln(1/r), r ~ U(0,1)
                r_energy = rand.random()
                A_t      = 4 * (1 - t / num_it) * math.log(1.0 / r_energy)

                # Generar candidatas en paralelo
                args_move = []
                for i in range(num_ind):
                    # Conejo aleatorio j ≠ i (para Fase 2)
                    candidates = list(range(num_ind))
                    candidates.remove(i)
                    j    = rand.choice(candidates)
                    seed = np.random.randint(0, 2**31)
                    args_move.append([X[i], X[j], A_t, t, num_it, seed])

                with Pool(processes=num_proc) as pool:
                    V = pool.starmap(
                        aro_move_rabbit, args_move,
                        chunksize=num_ind // num_proc + int(num_ind % num_proc != 0)
                    )

                # Corregir features sobrantes
                for i in range(num_ind):
                    V[i] = delete_features(V[i])

                # Evaluar candidatas en paralelo
                args_cost = [[V[i], ALPHA, BETA,
                              NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST,
                              TRAIN_X, TRAIN_Y, TEST_X, TEST_Y,
                              NUM_CHAR, MAX_FEATURES] for i in range(num_ind)]

                with Pool(processes=num_proc) as pool:
                    costs_v = pool.starmap(
                        cost_func, args_cost,
                        chunksize=num_ind // num_proc + int(num_ind % num_proc != 0)
                    )

                # Selección greedy: actualizar solo si mejora
                for i in range(num_ind):
                    if costs_v[i] < costs[i]:
                        X[i]     = V[i]
                        costs[i] = costs_v[i]

                # Actualizar mejor global
                current_best_idx = int(np.argmin(costs))
                if costs[current_best_idx] < best_cost:
                    best_solution = X[current_best_idx].copy()
                    best_cost     = costs[current_best_idx]

                curve[t - 1] = best_cost

                if not measure_mode:
                    print(f"Iteración {t} | Mejor coste = {best_cost:.6f}")

            return best_solution, curve
        
    # --------------------------------------------------------------------------
    #  EJECUCIÓN
    # --------------------------------------------------------------------------
    output_dir = "Measurements/" + alg_name

    if measure_mode:
        tracker = EmissionsTracker(measure_power_secs=999, output_dir=output_dir)
        tracker.start()
        t1 = time.time()

    try:

        if alg_name == "ABC":
            solution, best_solutions_fitness = abc()
        
        elif alg_name == "BOA":
            solution, best_solutions_fitness = boa()
        
        elif alg_name == "MFO":
            solution, best_solutions_fitness = mfo()
        
        elif alg_name == "HHO":
            solution, best_solutions_fitness = hho()
            
        elif alg_name == "ARO":
            solution, best_solutions_fitness = aro()

    finally:

        if measure_mode:
            t2 = time.time()
            tracker.stop()

        # Resultados finales — se ejecutan siempre aunque el algoritmo falle
        solution_fitness = best_solutions_fitness[-1]
        accuracy         = cost_to_acc(solution_fitness, solution)
        write_output(solution, accuracy)

        if measure_mode:
            write_time(t2 - t1, output_dir + "/times.csv")
            write_accuracy(accuracy,        output_dir + "/accuracy.csv")
            write_solution(solution,        output_dir + "/solutions.csv")
        else:
            print(f"Número de features de la mejor solución : {count_features(solution)}")
            print(f"Fitness de la mejor solución            = {solution_fitness}")
            print(f"Accuracy de la mejor solución           = {accuracy}")

            iterations_x = np.arange(1, num_it + 1)
            plt.plot(iterations_x, best_solutions_fitness)
            plt.title(alg_name + " - Iteración vs. Coste")
            plt.xlabel("Iteración")
            plt.ylabel("Coste")
            plt.show()