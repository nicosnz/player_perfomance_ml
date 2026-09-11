
import pandas as pd


COLUMNAS_PORTERO = [
    "Goals Against",
    "Goals against p 90",
    "Saves",
    "Saves %",
    "Clean Sheets",
    "% Clean sheets",
    "% Penalty saves",
    "Crosses Stopped"
]

COLUMNAS_DUPLICADAS = [
    "Goals Scored",
    "carries_prgc"
]

COLUMNAS_A_ELIMINAR = [
    "rk",
    "born",
    "nation",
    "squad",
    "comp"
]

COLUMNAS_CATEGORICAS = [
    "pos"
]


def separar_porteros(dataset):

    porteros = dataset[
        dataset["pos"] == "GK"
    ].reset_index(drop=True)

    jugadores_de_campo = dataset[
        dataset["pos"] != "GK"
    ].reset_index(drop=True)

    return jugadores_de_campo, porteros


def eliminar_columnas_portero(dataset):
    return dataset.drop(columns=COLUMNAS_PORTERO)


def eliminar_columnas_duplicadas(dataset):
    return dataset.drop(columns=COLUMNAS_DUPLICADAS)


def agregar_anio_inicio(dataset):

    dataset = dataset.copy()

    dataset["anio_inicio"] = (
        dataset["season"]
        .str.split("-")
        .str[0]
        .astype(int)
    )

    return dataset.drop(columns=["season"])


def eliminar_columnas_innecesarias(dataset):

    columnas = [
        c for c in COLUMNAS_A_ELIMINAR
        if c in dataset.columns
    ]

    return dataset.drop(columns=columnas)


def codificar_categoricas(dataset):

    columnas = [
        c for c in COLUMNAS_CATEGORICAS
        if c in dataset.columns and dataset[c].nunique() > 1
    ]

    return pd.get_dummies(dataset, columns=columnas, dtype=int)


def preparar_dataset_campo(dataset):

    jugadores_de_campo, porteros = separar_porteros(dataset)

    jugadores_de_campo = eliminar_columnas_portero(jugadores_de_campo)
    jugadores_de_campo = eliminar_columnas_duplicadas(jugadores_de_campo)

    porteros = eliminar_columnas_duplicadas(porteros)

    return jugadores_de_campo, porteros


def limpiar_dataset(dataset):

    dataset = agregar_anio_inicio(dataset)
    dataset = eliminar_columnas_innecesarias(dataset)
    dataset = codificar_categoricas(dataset)

    return dataset


# =========================
# Programa principal
# =========================

entrenamiento = pd.read_csv("datasets/dataset_entrenamiento.csv")
prediccion = pd.read_csv("datasets/dataset_prediccion.csv")

entrenamiento_campo, entrenamiento_porteros = preparar_dataset_campo(entrenamiento)
prediccion_campo, prediccion_porteros = preparar_dataset_campo(prediccion)

entrenamiento_campo = limpiar_dataset(entrenamiento_campo)
prediccion_campo = limpiar_dataset(prediccion_campo)
entrenamiento_porteros = limpiar_dataset(entrenamiento_porteros)
prediccion_porteros = limpiar_dataset(prediccion_porteros)

entrenamiento_campo.to_csv("datasets/dataset_entrenamiento_campo.csv", index=False)
entrenamiento_porteros.to_csv("datasets/dataset_entrenamiento_porteros.csv", index=False)
prediccion_campo.to_csv("datasets/dataset_prediccion_campo.csv", index=False)
prediccion_porteros.to_csv("datasets/dataset_prediccion_porteros.csv", index=False)

print("Entrenamiento - jugadores de campo:", entrenamiento_campo.shape)
print("Entrenamiento - porteros:", entrenamiento_porteros.shape)
print("Predicción - jugadores de campo:", prediccion_campo.shape)
print("Predicción - porteros:", prediccion_porteros.shape)
