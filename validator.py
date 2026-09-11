

def mostrar_informacion(dataset):

    print("===================================")
    print("INFORMACIÓN DEL DATASET")
    print("===================================")

    print("\nShape:")
    print(dataset.shape)

    print("\nRegistros por liga:")
    print(dataset["comp"].value_counts())

    print("\nRating nulos:")
    print(dataset["Rating"].isna().sum())

    print("\nPrimeros registros:")

    print(
        dataset[
            [
                "player",
                "squad",
                "pos",
                "season",
                "comp",
                "Goals",
                "Assists",
                "Rating"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


def comprobar_duplicados(dataset):

    duplicados = dataset.duplicated(
        subset=["player", "season", "comp"]
    ).sum()

    print("\nDuplicados jugador + temporada + liga:")
    print(duplicados)