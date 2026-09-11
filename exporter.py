

def guardar_dataset(dataset, ruta):

    dataset.to_csv(
        ruta,
        index=False
    )

    print(f"\nDataset guardado en: {ruta}")