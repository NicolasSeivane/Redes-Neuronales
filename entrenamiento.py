import numpy as np
import random as rd
import json

# ===== Datasets =====
datasets = {
    "Dataset1": {
        "X": np.array([[1.1946,3.9502],[0.8788,1.6595],[1.1907,1.6117],[1.4180,3.8272],[0.2032,1.9208],[2.7571,1.9931],[4.7125,2.8166],[3.9392,1.1032],[1.2072,0.6123],[3.4799,1.9982],[0.4763,0.1020]]),
        "y": np.array([1,1,1,1,1,-1,-1,-1,-1,-1,-1])
    },
    "Dataset2": {
        "X": np.array([[1.1946,5.9502],[0.8788,3.6595],[1.1907,3.6117],[1.4180,5.8272],[0.2032,3.9208],[2.7571,3.9931],[4.7125,4.8166],[3.9392,3.1032],[1.2072,2.6123],[3.4799,3.9982],[0.4763,2.1020]]),
        "y": np.array([1,1,1,1,1,-1,-1,-1,-1,-1,-1])
    },
    "Dataset3": {
        "X": np.array([[2.6946,6.9502],[2.3788,4.6595],[2.6907,4.6117],[2.9180,6.8272],[1.7032,4.9208],[4.2571,4.9931],[6.2125,5.8166],[5.4392,4.1032],[2.7072,3.6123],[4.9799,4.9982],[1.9763,3.1020]]),
        "y": np.array([1,1,1,1,1,-1,-1,-1,-1,-1,-1])
    },
    "Dataset4": {
        "X": np.array([[2.6946,5.9502],[2.3788,3.6595],[2.6907,3.6117],[2.9180,5.8272],[1.7032,3.9208],[4.2571,1.9931],[6.2125,1.8166],[5.4392,0.1032],[2.7072,2.6123],[4.9799,1.9982],[1.9763,2.1020]]),
        "y": np.array([1,1,1,1,1,-1,-1,-1,-1,-1,-1])
    }
}

# ===== Funciones =====
def signo(x):
    return 1 if x>0 else -1

def calcularError(X, y, w, b):
    O_total = np.sign(np.dot(X, w) + b)
    return float(np.sum(O_total != y))

def perceptron_simple(X, y, COTA=300, lr=0.1, usar_sesgo=True):
    p = X.shape[0]
    sesgo = 0.0 if usar_sesgo else 0.0  # Inicializa el sesgo en 0.0
    w = np.zeros(X.shape[1])  # Inicializa los pesos en cero

    listas_pesos = []
    listas_sesgos = []
    listas_error = []

    error_min = np.inf
    idx_best = -1

    for i in range(COTA):
        indice = rd.randint(0, p-1)
        h = np.dot(X[indice], w) + sesgo
        O = signo(h)
        w_nuevo = lr * (y[indice] - O) * X[indice]
        w = w + w_nuevo
        if usar_sesgo:
            sesgo = sesgo + lr * (y[indice] - O)

        error = calcularError(X, y, w, sesgo)
        
        listas_pesos.append(w.copy())
        listas_sesgos.append(sesgo)
        listas_error.append(error)

        if error < error_min:
            error_min = error
            idx_best = i

        if error == 0:
            error_min = 0
            idx_best = i
            break

    if idx_best >= 0:
        w_best = listas_pesos[idx_best]
        b_best = listas_sesgos[idx_best]
    else:
        w_best = w.copy()
        b_best = sesgo

    # Solo guardar hasta el mejor índice (error mínimo)
    listas_pesos = listas_pesos[:idx_best+1]
    listas_sesgos = listas_sesgos[:idx_best+1]
    listas_error = listas_error[:idx_best+1]

    return listas_pesos, listas_sesgos, listas_error, w_best, b_best, error_min

# ===== Entrenar para varios learning rates y con/sin sesgo =====
learning_rates = [0.1, 0.01, 0.001]
resultados = {}

for d_name, data in datasets.items():
    resultados[d_name] = {"con": [], "sin": []}
    for lr in learning_rates:
        # Con sesgo
        p, b, e, w_b, b_b, err = perceptron_simple(data["X"], data["y"], COTA=100, lr=lr, usar_sesgo=True)
        resultados[d_name]["con"].append({
            "pesos": p,
            "sesgos": b,
            "errores": e,
            "w_best": w_b,
            "b_best": b_b,
            "error_best": err
        })
        # Sin sesgo
        p, b, e, w_b, b_b, err = perceptron_simple(data["X"], data["y"], COTA=100, lr=lr, usar_sesgo=False)
        resultados[d_name]["sin"].append({
            "pesos": p,
            "sesgos": b,
            "errores": e,
            "w_best": w_b,
            "b_best": b_b,
            "error_best": err
        })

# Función para convertir arrays de numpy a listas normales (para JSON)
def convertir(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.integer)):
        return float(obj)
    return obj

# Guardar resultados en JSON
with open("resultados_entrenamiento.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2, default=convertir)