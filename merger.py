
import pandas as pd


def crear_dataset_liga(fbref, liga, ruta_sofascore):

    fbref_liga = fbref[fbref["comp"] == liga]

    sofascore = pd.read_csv(ruta_sofascore)

    sofascore_rating = sofascore[
        ["Player Name", "Rating"]
    ]

    dataset = fbref_liga.merge(
        sofascore_rating,
        left_on="player",
        right_on="Player Name",
        how="inner"
    )

    dataset = dataset.drop(
        columns=["Player Name"]
    )

    return dataset


def crear_dataset_temporada(fbref, archivos_sofascore):

    datasets = []

    for liga, ruta in archivos_sofascore.items():

        dataset_liga = crear_dataset_liga(
            fbref,
            liga,
            ruta
        )

        datasets.append(dataset_liga)

    return pd.concat(
        datasets,
        ignore_index=True
    )
    
def unir_temporadas(datasets):
    return pd.concat(
        datasets,
        ignore_index=True
    )