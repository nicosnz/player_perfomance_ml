
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


TARGET = "rating_siguiente_temporada"

UMBRAL_MULTICOLINEALIDAD = 0.85
UMBRAL_BAJA_RELACION = 0.05


def separar_columnas(dataset):

    categoricas = dataset.select_dtypes(
        include="object"
    ).columns.tolist()

    numericas = dataset.select_dtypes(
        include="number"
    ).columns.tolist()

    return categoricas, numericas


def calcular_matriz_correlacion(dataset, numericas):
    return dataset[numericas].corr()


def guardar_heatmap(matriz, ruta):

    plt.figure(figsize=(28, 24))

    sns.heatmap(
        matriz,
        cmap="coolwarm",
        center=0,
        annot=False,
        square=True
    )

    plt.title("Matriz de correlación - dataset_entrenamiento")
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()


def correlacion_con_target(matriz, target):

    return (
        matriz[target]
        .drop(target)
        .sort_values(key=abs, ascending=False)
    )


def pares_multicolineales(matriz, umbral, target):

    pares = []
    columnas = matriz.columns

    for i in range(len(columnas)):
        for j in range(i + 1, len(columnas)):

            col_i, col_j = columnas[i], columnas[j]

            if col_i == target or col_j == target:
                continue

            valor = matriz.iloc[i, j]

            if abs(valor) >= umbral:
                pares.append((col_i, col_j, valor))

    return sorted(
        pares,
        key=lambda x: abs(x[2]),
        reverse=True
    )


def imprimir_reporte(dataset, matriz, categoricas):

    corr_target = correlacion_con_target(matriz, TARGET)
    multicolineales = pares_multicolineales(
        matriz,
        UMBRAL_MULTICOLINEALIDAD,
        TARGET
    )
    baja_relacion = corr_target[
        corr_target.abs() < UMBRAL_BAJA_RELACION
    ]

    print("=" * 60)
    print("INFORME DE CORRELACIÓN")
    print("=" * 60)

    print(f"\nFilas: {dataset.shape[0]}  Columnas: {dataset.shape[1]}")
    print(f"Columnas numéricas analizadas: {len(matriz.columns)}")

    print("\n--- Columnas categóricas (no entran en la matriz) ---")
    for col in categoricas:
        print(f"  {col} ({dataset[col].nunique()} valores únicos)")

    print(
        f"\n--- Correlación con el target "
        f"'{TARGET}' (ordenado por magnitud) ---"
    )
    print(corr_target.to_string())

    print(
        f"\n--- Features con relación muy débil con el target "
        f"(|r| < {UMBRAL_BAJA_RELACION}) ---"
    )
    if baja_relacion.empty:
        print("  Ninguna")
    else:
        print(baja_relacion.to_string())

    print(
        f"\n--- Pares con posible multicolinealidad "
        f"(|r| >= {UMBRAL_MULTICOLINEALIDAD}) ---"
    )
    if not multicolineales:
        print("  Ninguno")
    else:
        for col_i, col_j, valor in multicolineales:
            print(f"  {col_i}  <->  {col_j}   r = {valor:.3f}")


# =========================
# Programa principal
# =========================

dataset = pd.read_csv("datasets/dataset_entrenamiento_campo.csv")

categoricas, numericas = separar_columnas(dataset)

matriz = calcular_matriz_correlacion(dataset, numericas)

guardar_heatmap(
    matriz,
    "diagramas/correlacion_entrenamiento_campo.png"
)

imprimir_reporte(dataset, matriz, categoricas)
