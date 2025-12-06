# streamlit_app.py
import numpy as np
import pandas as pd
import streamlit as st

from perceptron import min_max_normalize, entrenar_perceptron, activation

st.set_page_config(page_title="Perceptrón Dinámico", layout="wide")

st.title("🧠 Perceptrón Dinámico con Normalización Min–Max")
st.write(
    """
Interfaz gráfica para jugar con un perceptrón de **cualquier número de características (x₁...xₙ)**  
Usa el módulo `perceptron.py` (entrenamiento por épocas) con una UI mucho más cómoda.
"""
)

# ===========================================================
# SECCIÓN 1: DEFINICIÓN DEL CONJUNTO DE ENTRENAMIENTO
# ===========================================================
st.header("Datos de entrenamiento")

col_n, col_hint = st.columns([1, 3])
with col_n:
    n_features = st.number_input(
        "Número de características (X) por patrón",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
    )

with col_hint:
    st.info(
        "Cada fila será un patrón de entrenamiento. Las columnas `x1...xn` son las características "
        "y la columna `y` es la salida deseada (0 o 1). Puedes agregar/eliminar filas."
    )

# Crear/actualizar DataFrame en sesión
if "df_train" not in st.session_state or st.session_state.get("df_train_n") != n_features:
    cols = [f"x{i+1}" for i in range(n_features)] + ["y"]
    data = [
        [0.0] * n_features + [0],
        [1.0] * n_features + [1],
    ]
    st.session_state.df_train = pd.DataFrame(data, columns=cols)
    st.session_state.df_train_n = n_features

df_train = st.data_editor(
    st.session_state.df_train,
    num_rows="dynamic",
    use_container_width=True,
    key="data_editor_train",
)
st.session_state.df_train = df_train  # guardar cambios


def preparar_datos(df, n_features: int):
    """Convierte el DataFrame a X (float) e y (int), validando NaNs."""
    if df.empty:
        st.error("⚠ Debes tener al menos un patrón de entrenamiento.")
        return None, None

    required_cols = [f"x{i+1}" for i in range(n_features)] + ["y"]
    for c in required_cols:
        if c not in df.columns:
            st.error(f"Falta la columna `{c}` en la tabla de datos.")
            return None, None

    # Eliminar filas completamente vacías
    df_clean = df.dropna(how="all")
    if df_clean.empty:
        st.error("⚠ Todos los patrones están vacíos.")
        return None, None

    # Verificar NaNs en las columnas necesarias
    if df_clean[required_cols].isna().any().any():
        st.error("⚠ No puede haber valores vacíos (NaN) en X o en y.")
        return None, None

    X = df_clean[[f"x{i+1}" for i in range(n_features)]].astype(float).to_numpy()
    y = df_clean["y"].astype(int).to_numpy()

    # Verificar que y solo contenga 0 o 1
    if not np.isin(y, [0, 1]).all():
        st.error("⚠ La columna y solo puede contener 0 o 1.")
        return None, None

    return X, y


# ===========================================================
# SECCIÓN 2: HIPERPARÁMETROS DEL PERCEPTRÓN
# ===========================================================
st.header("Parámetros del modelo")

col_left, col_right = st.columns(2)

with col_left:
    init_mode = st.selectbox(
        "Inicialización de pesos",
        options=["Ceros", "Aleatorios N(0,1)"],
        index=0,
    )

    b_init = st.number_input("Sesgo inicial (b)", value=0.0, step=0.1)

with col_right:
    eta = st.number_input(
        "Tasa de aprendizaje (η)",
        min_value=0.0001,
        max_value=1.0,
        value=0.1,
        step=0.01,
    )
    max_iter = st.number_input(
        "Máximo de iteraciones (épocas)",
        min_value=1,
        max_value=10_000,
        value=20,
        step=1,
    )

# ===========================================================
# SECCIÓN 3: ENTRENAR EL MODELO
# ===========================================================
st.header("Entrenar perceptrón")

train_button = st.button("🚀 Entrenar", type="primary")

if train_button:
    X_raw, y = preparar_datos(df_train, n_features)
    if X_raw is not None:
        # Normalizar
        X_norm, X_min, X_max = min_max_normalize(X_raw)
        X_norm = np.round(X_norm, 2)

        st.subheader("Datos normalizados (Min–Max)")
        st.dataframe(
            pd.DataFrame(
                X_norm,
                columns=[f"x{i+1}" for i in range(n_features)],
            ),
            use_container_width=True,
        )

        # Inicializar pesos
        if init_mode == "Ceros":
            w_init = np.zeros(n_features, dtype=float)
        else:
            w_init = np.random.randn(n_features).astype(float)

        st.write("Pesos iniciales:", w_init)
        st.write("Sesgo inicial:", b_init)

        # Entrenar usando la función de tu módulo
        with st.spinner("Entrenando perceptrón..."):
            w_final, b_final, historial = entrenar_perceptron(
                X_norm, y, w_init, b_init, eta, int(max_iter)
            )

        # Guardar en sesión para poder validar luego
        st.session_state.trained = True
        st.session_state.X_min = X_min
        st.session_state.X_max = X_max
        st.session_state.w_final = w_final
        st.session_state.b_final = b_final
        st.session_state.historial = historial
        st.session_state.n_features = n_features

        st.success("✅ Entrenamiento completado")

        # ---------- RESUMEN POR ÉPOCA ----------
        st.subheader("Resumen por iteración (época)")
        resumen_data = []
        for h in historial:
            resumen_data.append(
                {
                    "Iteración": h["iter"],
                    "Errores": h["errores"],
                    "w": np.round(h["w"], 3),
                    "b": round(h["b"], 3),
                }
            )
        st.dataframe(pd.DataFrame(resumen_data), use_container_width=True)

        # Evolución de errores
        st.subheader("Evolución de errores por iteración")
        errores_por_iter = pd.DataFrame(
            {
                "iter": [h["iter"] for h in historial],
                "errores": [h["errores"] for h in historial],
            }
        ).set_index("iter")
        st.line_chart(errores_por_iter)

        # ---------- TABLA DETALLADA PARA EL INFORME ----------
        st.subheader("Tabla detallada (para el informe)")

        # Construimos una sola tabla con TODAS las actualizaciones
        filas = []
        for h in historial:
            iter_idx = h["iter"]
            for r in h["registros"]:
                fila = {
                    "Iteración": iter_idx,
                    "Patrón": r["patron"],
                    "z": r["z"],
                    "ŷ": r["pred"],
                    "Error": r["error"],
                    "b": r["b"],
                }
                # Expandir vector de pesos en columnas w1...wn
                w_vec = np.array(r["w"], dtype=float)
                for j, val in enumerate(w_vec):
                    fila[f"w{j+1}"] = val
                filas.append(fila)

        df_full_log = pd.DataFrame(filas)

        st.dataframe(df_full_log, use_container_width=True)

        # Botón para descargar como CSV (para pegar en Excel/Word/LaTeX)
        csv_bytes = df_full_log.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Descargar tabla completa (CSV)",
            data=csv_bytes,
            file_name="perceptron_entrenamiento_detalle.csv",
            mime="text/csv",
        )

        # Detalle por época en un expander (si querés verlo separado)
        with st.expander("🔎 Ver detalle por época"):
            for h in historial:
                st.markdown(f"#### Iteración {h['iter']}")
                filas_epoch = []
                for r in h["registros"]:
                    fila = {
                        "Patrón": r["patron"],
                        "z": r["z"],
                        "ŷ": r["pred"],
                        "Error": r["error"],
                        "b": r["b"],
                    }
                    w_vec = np.array(r["w"], dtype=float)
                    for j, val in enumerate(w_vec):
                        fila[f"w{j+1}"] = val
                    filas_epoch.append(fila)
                df_regs = pd.DataFrame(filas_epoch)
                st.dataframe(df_regs, use_container_width=True)

        st.subheader("Parámetros finales del modelo")
        st.write("w_final:", np.round(w_final, 3))
        st.write("b_final:", round(b_final, 3))

# ===========================================================
# SECCIÓN 4: VALIDACIÓN DE NUEVOS DATOS
# ===========================================================
st.header("Validar nuevos datos")

if not st.session_state.get("trained", False):
    st.info("⚠ Entrena el modelo primero para poder validar nuevos datos.")
else:
    X_min = st.session_state.X_min
    X_max = st.session_state.X_max
    w_final = st.session_state.w_final
    b_final = st.session_state.b_final
    n_features = st.session_state.n_features

    st.write(
        "Introduce un nuevo vector x = (x₁...xₙ) en la **escala original**. "
        "Se normalizará automáticamente usando los min/max del conjunto de entrenamiento."
    )

    cols_inputs = st.columns(n_features)
    x_new = np.zeros(n_features, dtype=float)
    for i in range(n_features):
        with cols_inputs[i]:
            x_new[i] = st.number_input(
                f"x{i+1} (nuevo)",
                value=0.0,
                key=f"x_new_{i}",
            )

    if st.button("📌 Validar nuevo patrón"):
        # Normalizar usando min y max originales
        x_norm = (x_new - X_min) / (X_max - X_min)
        x_norm = np.round(x_norm, 2)

        z = np.dot(w_final, x_norm) + b_final
        pred = activation(z)

        st.write("Vector original:", x_new)
        st.write("Vector normalizado:", x_norm)
        st.write(f"z = {z:.4f}")
        st.success(f"Predicción del perceptrón: **{pred}** (0 = rechazo, 1 = aceptación)")
