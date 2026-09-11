
import pandas as pd


def crear_mapa_rating(dataset):

    return (
        dataset
        .groupby(["player", "born", "season"])["Rating"]
        .mean()
        .to_dict()
    )


def obtener_siguiente_temporada(season, temporadas_ordenadas):

    indice = temporadas_ordenadas.index(season)

    if indice + 1 >= len(temporadas_ordenadas):
        return None

    return temporadas_ordenadas[indice + 1]


def agregar_rating_siguiente_temporada(dataset):

    temporadas_ordenadas = sorted(
        dataset["season"].unique()
    )

    mapa_rating = crear_mapa_rating(dataset)

    dataset = dataset.copy()

    def buscar_rating_siguiente(fila):

        siguiente = obtener_siguiente_temporada(
            fila["season"],
            temporadas_ordenadas
        )

        if siguiente is None:
            return None

        return mapa_rating.get(
            (fila["player"], fila["born"], siguiente)
        )

    dataset["rating_siguiente_temporada"] = dataset.apply(
        buscar_rating_siguiente,
        axis=1
    )

    return dataset


def separar_entrenamiento_prediccion(dataset):

    entrenamiento = dataset[
        dataset["rating_siguiente_temporada"].notna()
    ].reset_index(drop=True)

    prediccion = dataset[
        dataset["rating_siguiente_temporada"].isna()
    ].reset_index(drop=True)

    return entrenamiento, prediccion
