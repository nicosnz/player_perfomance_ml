from config import TEMPORADAS
import pandas as pd
from loader import cargar_csv
from merger import crear_dataset_temporada,unir_temporadas
from validator import (
    mostrar_informacion,
    comprobar_duplicados,
    eliminar_duplicados
)
from target import (
    agregar_rating_siguiente_temporada,
    separar_entrenamiento_prediccion
)
from exporter import guardar_dataset


def procesar_temporada(nombre_temporada):

    configuracion = TEMPORADAS[nombre_temporada]

    # =========================
    # Cargar FBref
    # =========================

    fbref = cargar_csv(
        configuracion["fbref"]
    )

    # =========================
    # Crear dataset
    # =========================

    dataset = crear_dataset_temporada(
        fbref,
        configuracion["sofascore"]
    )

    # =========================
    # Validar
    # =========================

    # mostrar_informacion(dataset)

    # comprobar_duplicados(dataset)

    return dataset


# =========================
# Programa principal
# =========================

datasets_por_temporada = [
    procesar_temporada(nombre_temporada)
    for nombre_temporada in TEMPORADAS
]


# ===================================
# UNIR TODAS LAS TEMPORADAS
# ===================================

dataset_historico = unir_temporadas(datasets_por_temporada)


# print("\nAntes de eliminar duplicados:")
# print(dataset_historico.shape)

dataset_historico = eliminar_duplicados(
    dataset_historico
)

# print("\nDespués de eliminar duplicados:")
# print(dataset_historico.shape)

# comprobar_duplicados(dataset_historico)

# ===================================
# TARGET: RATING DE LA TEMPORADA SIGUIENTE
# ===================================

dataset_historico = agregar_rating_siguiente_temporada(
    dataset_historico
)

guardar_dataset(
    dataset_historico,
    "datasets/dataset_historico.csv"
)

dataset_entrenamiento, dataset_prediccion = separar_entrenamiento_prediccion(
    dataset_historico
)

guardar_dataset(
    dataset_entrenamiento,
    "datasets/dataset_entrenamiento.csv"
)

guardar_dataset(
    dataset_prediccion,
    "datasets/dataset_prediccion.csv"
)