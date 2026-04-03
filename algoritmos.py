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
#  Ecuación de actualización (Karaboga & Basturk, 2007):
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
    #  Referencia: Karaboga & Basturk (2007), J. Global Optimization 39:459-471
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
    # --------------------------------------------------------------------------
    #  BLOQUE DE EJECUCIÓN DEL ALGORITMO
    #
    #  Aquí se irán añadiendo los algoritmos poco a poco.
    #  Cada algoritmo debe producir al final:
    #    - solution             : vector binario [MAX_FEATURES]
    #    - best_solutions_fitness : vector [num_it] con el mejor cost por iteración
    # --------------------------------------------------------------------------

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