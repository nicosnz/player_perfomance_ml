

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
        subset=["player", "born", "season", "comp"]
    ).sum()

    print("\nDuplicados jugador + temporada + liga:")
    print(duplicados)
def mostrar_detalle_duplicados(dataset):

    duplicados = dataset[
        dataset.duplicated(
            subset=["player", "born", "season", "comp"],
            keep=False
        )
    ]

    duplicados = duplicados.sort_values(
        ["season", "comp", "player"]
    )

    print("\n===================================")
    print("DETALLE DE DUPLICADOS")
    print("===================================")

    print(
        duplicados[
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
        ].to_string(index=False)
    )

    return duplicados
def eliminar_duplicados(dataset):

    dataset_limpio = dataset.drop_duplicates(
        subset=["player", "born", "season", "comp"],
        keep="first"
    )

    return dataset_limpio