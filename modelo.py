
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET = "rating_siguiente_temporada"
COLUMNAS_NO_FEATURE = ["player", TARGET]


def cargar_datos():
    return pd.read_csv("datasets/dataset_entrenamiento_campo.csv")


def agregar_features_derivadas(X):

    X = X.copy()

    # no lineal: la curva de rendimiento por edad no es una recta
    X["age_squared"] = X["age"] ** 2

    # interaccion: el peso del rating actual podria no ser constante en el tiempo
    X["rating_x_anio"] = X["Rating"] * X["anio_inicio"]

    # log: goles/asistencias tienen cola larga (pocos jugadores con 20+),
    # log1p comprime esa cola en vez de dejar que unos pocos outliers
    # dominen la recta
    X["log_goals"] = np.log1p(X["Goals"])
    X["log_assists"] = np.log1p(X["Assists"])
    X = X.drop(columns=["Goals", "Assists"])

    return X


def separar_x_y(dataset):

    columnas_feature = [
        c for c in dataset.columns
        if c not in COLUMNAS_NO_FEATURE
    ]

    X = agregar_features_derivadas(dataset[columnas_feature])
    y = dataset[TARGET]

    return X, y


def calcular_metricas(y_real, y_pred):

    return {
        "MAE": mean_absolute_error(y_real, y_pred),
        "RMSE": mean_squared_error(y_real, y_pred) ** 0.5,
        "R2": r2_score(y_real, y_pred)
    }


def imprimir_metricas(nombre, metricas):
    print(
        f"{nombre:30s}  MAE={metricas['MAE']:.4f}  "
        f"RMSE={metricas['RMSE']:.4f}  R2={metricas['R2']:.4f}"
    )


def evaluar_baseline_walk_forward(dataset):

    anios = sorted(dataset["anio_inicio"].unique())
    filas = []

    for anio_test in anios[1:]:

        test = dataset[dataset["anio_inicio"] == anio_test]
        train = dataset[dataset["anio_inicio"] < anio_test]

        pred_rating = test["Rating"].values
        pred_promedio = np.full(len(test), train[TARGET].mean())

        m_rating = calcular_metricas(test[TARGET], pred_rating)
        m_promedio = calcular_metricas(test[TARGET], pred_promedio)

        filas.append({"anio_test": anio_test, "modelo": "Baseline (rating actual)", **m_rating})
        filas.append({"anio_test": anio_test, "modelo": "Baseline (promedio train)", **m_promedio})

    return pd.DataFrame(filas)


def evaluar_modelo_walk_forward(dataset, nombre, constructor_modelo):

    anios = sorted(dataset["anio_inicio"].unique())
    filas = []

    for anio_test in anios[1:]:

        train = dataset[dataset["anio_inicio"] < anio_test]
        test = dataset[dataset["anio_inicio"] == anio_test]

        X_train, y_train = separar_x_y(train)
        X_test, y_test = separar_x_y(test)

        scaler = StandardScaler()
        X_train_esc = scaler.fit_transform(X_train)
        X_test_esc = scaler.transform(X_test)

        modelo = constructor_modelo()
        modelo.fit(X_train_esc, y_train)
        pred = modelo.predict(X_test_esc)

        metricas = calcular_metricas(y_test, pred)
        metricas["anio_test"] = anio_test
        metricas["modelo"] = nombre
        metricas["n_train"] = len(train)
        metricas["n_test"] = len(test)

        filas.append(metricas)

    return pd.DataFrame(filas)


def resumen(df_resultados, nombre):

    fila = df_resultados[df_resultados["modelo"] == nombre]

    return {
        "MAE": fila["MAE"].mean(),
        "RMSE": fila["RMSE"].mean(),
        "R2": fila["R2"].mean()
    }


# =========================
# Programa principal
# =========================

dataset = cargar_datos()

anios = sorted(dataset["anio_inicio"].unique())
print(f"Temporadas de origen disponibles: {anios}")
print(f"Folds walk-forward (test = cada año, train = todos los anteriores): {anios[1:]}")
print()

resultados_baseline = evaluar_baseline_walk_forward(dataset)

modelos = {
    "Regresión lineal (OLS)": lambda: LinearRegression(),
    "Ridge (L2)": lambda: RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5),
    "Lasso (L1)": lambda: LassoCV(cv=5, random_state=42, max_iter=20000),
    "ElasticNet (L1+L2)": lambda: ElasticNetCV(
        l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1],
        cv=5, random_state=42, max_iter=20000
    ),
}

resultados_por_modelo = {}

for nombre, constructor in modelos.items():
    resultados_por_modelo[nombre] = evaluar_modelo_walk_forward(dataset, nombre, constructor)

print("=" * 78)
print("WALK-FORWARD POR FOLD (detalle)")
print("=" * 78)

for nombre, df in resultados_por_modelo.items():
    print(f"\n--- {nombre} ---")
    print(df[["anio_test", "n_train", "n_test", "MAE", "RMSE", "R2"]].to_string(index=False))

print()
print("=" * 78)
print("PROMEDIO WALK-FORWARD (5 folds, esto es lo que importa)")
print("=" * 78)
imprimir_metricas("Baseline (rating actual)", resumen(resultados_baseline, "Baseline (rating actual)"))
imprimir_metricas("Baseline (promedio train)", resumen(resultados_baseline, "Baseline (promedio train)"))
for nombre, df in resultados_por_modelo.items():
    imprimir_metricas(nombre, resumen(df, nombre))

# =========================
# Modelo final: se re-entrena con TODOS los datos disponibles
# (usando el mejor tipo de regularización encontrado arriba)
# =========================

X_full, y_full = separar_x_y(dataset)
scaler_full = StandardScaler()
X_full_esc = scaler_full.fit_transform(X_full)

modelo_final = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
modelo_final.fit(X_full_esc, y_full)

coeficientes = pd.Series(
    modelo_final.coef_,
    index=X_full.columns
).sort_values(key=abs, ascending=False)

print()
print("=" * 78)
print(f"MODELO FINAL: Ridge (alpha={modelo_final.alpha_:.4f}), entrenado con las {len(dataset)} filas")
print("=" * 78)
print("\nTop 15 coeficientes (features estandarizadas, magnitud = importancia relativa):")
print(coeficientes.head(15).to_string())
