import numpy as np

# ===========================================================
# FUNCIÓN DE NORMALIZACIÓN MIN–MAX (para cualquier n)
# ===========================================================
def min_max_normalize(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min)
    return X_norm, X_min, X_max


# ===========================================================
# ACTIVACIÓN DEL PERCEPTRÓN
# ===========================================================
def activation(z):
    return 1 if z >= 0 else 0


# ===========================================================
# INGRESO DINÁMICO DE DATOS
# (cualquier cantidad de variables x1...xn)
# ===========================================================
def ingresar_datos():
    print("\n=== INGRESO DE DATOS DEL PERCEPTRÓN ===")
    
    n = int(input("¿Cuántas características (X) tendrá cada patrón?: "))
    X = []
    y = []

    while True:
        continuar = input("\n¿Desea ingresar un nuevo patrón? (s/n): ").lower()
        if continuar == "n":
            break

        print(f"Ingrese los {n} valores del patrón:")
        x = [float(input(f"  x{i+1}: ")) for i in range(n)]

        salida = int(input("Valor deseado (y = 0 o 1): "))

        X.append(x)
        y.append(salida)

    return np.array(X, dtype=float), np.array(y, dtype=int), n


# ===========================================================
# ENTRENAMIENTO GENERAL DEL PERCEPTRÓN
# ===========================================================
def entrenar_perceptron(X, y, w, b, eta, max_iter):
    historial = []

    for epoch in range(1, max_iter + 1):
        print(f"\n=========== ITERACIÓN {epoch} ===========")
        errores_epoch = 0
        registros = []

        for i in range(len(X)):
            x_i = X[i]
            y_real = y[i]

            z = np.dot(w, x_i) + b
            y_pred = activation(z)
            error = y_real - y_pred

            # Actualización
            w = w + eta * error * x_i
            b = b + eta * error

            # ► Registro detallado de CADA actualización:
            #    incluye z, predicción, error, pesos y sesgo
            registros.append({
                "patron": i + 1,
                "z": z,
                "pred": y_pred,
                "error": error,
                "w": w.copy(),
                "b": b
            })

            print(f"Patrón {i+1}: z={z:.4f}, pred={y_pred}, error={error}")
            print(f"   w={np.round(w,3)}, b={round(b,3)}")

            if error != 0:
                errores_epoch += 1

        # Registro por época (resumen)
        historial.append({
            "iter": epoch,
            "w": w.copy(),
            "b": b,
            "errores": errores_epoch,
            "registros": registros
        })

        if errores_epoch == 0:
            print("\n>>> EL MODELO CONVERGIÓ <<<")
            break

    return w, b, historial


# ===========================================================
# VALIDACIÓN DEL MODELO (cualquier n)
# ===========================================================
def validar_modelo(X_min, X_max, w, b):
    print("\n===== VALIDACIÓN DE NUEVOS DATOS =====")

    n = len(w)

    while True:
        cont = input("\n¿Desea validar un nuevo dato? (s/n): ").lower()
        if cont == "n":
            break

        print(f"Ingrese {n} valores para el nuevo dato:")
        x = np.array([float(input(f"x{i+1}: ")) for i in range(n)], dtype=float)

        # Normalizar usando min y max originales
        x_norm = (x - X_min) / (X_max - X_min)
        x_norm = np.round(x_norm, 2)

        z = np.dot(w, x_norm) + b
        pred = activation(z)

        print(f"Normalizado: {x_norm}, z={z:.3f}, predicción={pred}")


# ===========================================================
# PROGRAMA PRINCIPAL
# ===========================================================
def main():
    print("=== PERCEPTRÓN DINÁMICO CON NORMALIZACIÓN ===")

    # 1. Ingreso de datos
    X_raw, y, n = ingresar_datos()

    # 2. Normalizar
    X_norm, X_min, X_max = min_max_normalize(X_raw)
    X_norm = np.round(X_norm, 2)

    print("\nDatos normalizados:")
    print(X_norm)

    # 3. Pedir pesos iniciales dinámicos
    print("\n=== PARÁMETROS DEL MODELO ===")
    print(f"Debe ingresar {n} pesos iniciales (para x1 ... x{n})")

    w = np.array([float(input(f"Peso w{i+1}: ")) for i in range(n)], dtype=float)
    b = float(input("Sesgo inicial b: "))
    eta = float(input("Tasa de aprendizaje (eta): "))
    max_iter = int(input("Máximo de iteraciones: "))

    # 4. Entrenar
    w_final, b_final, historial = entrenar_perceptron(
        X_norm, y, w, b, eta, max_iter
    )

    # 5. Mostrar resumen por iteración (época)
    print("\n===== EVOLUCIÓN POR ITERACIÓN =====")
    for h in historial:
        print(f"Iter {h['iter']}: w={np.round(h['w'],3)}, b={h['b']}, errores={h['errores']}")

    # 6. Validación
    validar_modelo(X_min, X_max, w_final, b_final)

    print("\n=== FIN DEL PROGRAMA ===")


# Ejecutar
if __name__ == "__main__":
    main()
