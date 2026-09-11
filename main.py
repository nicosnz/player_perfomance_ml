# main.py

from config import TEMPORADAS
from loader import cargar_csv
from merger import crear_dataset_temporada
from validator import (
    mostrar_informacion,
    comprobar_duplicados
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

    mostrar_informacion(dataset)

    comprobar_duplicados(dataset)

    # =========================
    # Guardar
    # =========================

    ruta_salida = (
        f"datasets/dataset_{nombre_temporada}.csv"
    )

    guardar_dataset(
        dataset,
        ruta_salida
    )

    return dataset


# =========================
# Programa principal
# =========================

dataset_2022_23 = procesar_temporada(
    "2022-23"
)

dataset_2023_24 = procesar_temporada(
    "2023-24"
)