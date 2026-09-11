import pandas as pd

# =========================
# Cargar FBref
# =========================

fbref = pd.read_csv("datasets/FBref/cleaned_2023-24.csv")


# =========================
# Función para unir una liga
# =========================

def crear_dataset_liga(fbref, liga, archivo_sofascore):

    # Filtrar liga en FBref
    fbref_liga = fbref[fbref["comp"] == liga]

    # Cargar SofaScore
    sofascore = pd.read_csv(archivo_sofascore)

    # Quedarnos solamente con nombre y rating
    sofascore_rating = sofascore[["Player Name", "Rating"]]

    # Unir por nombre exacto
    dataset = fbref_liga.merge(
        sofascore_rating,
        left_on="player",
        right_on="Player Name",
        how="inner"
    )

    # Eliminar nombre duplicado de SofaScore
    dataset = dataset.drop(columns=["Player Name"])

    return dataset


# =========================
# Crear datasets
# =========================

bundesliga = crear_dataset_liga(
    fbref,
    "Bundesliga",
    "datasets/sofascore/2023-2024/Bundesliga20232024.csv"
)

laliga = crear_dataset_liga(
    fbref,
    "La Liga",
    "datasets/sofascore/2023-2024/laliga20232024.csv"
)

serie_a = crear_dataset_liga(
    fbref,
    "Serie A",
    "datasets/sofascore/2023-2024/SerieA20232024.csv"
)

ligue_1 = crear_dataset_liga(
    fbref,
    "Ligue 1",
    "datasets/sofascore/2023-2024/ligue120232024.csv"
)

premier_league = crear_dataset_liga(
    fbref,
    "Premier League",
    "datasets/sofascore/2023-2024/PL20232024.csv"
)


# =========================
# Unir todas las ligas
# =========================

dataset_global = pd.concat(
    [
        bundesliga,
        laliga,
        serie_a,
        ligue_1,
        premier_league
    ],
    ignore_index=True
)


# =========================
# Mostrar información
# =========================

print("Shape:", dataset_global.shape)

print("\nJugadores por liga:")
print(dataset_global["comp"].value_counts())

print("\nPrimeros registros:")
print(
    dataset_global[
        ["player", "squad", "pos", "season", "comp", "Goals", "Assists", "Rating"]
    ].head(20).to_string(index=False)
)


# =========================
# Comprobar valores nulos
# =========================

print("\nValores nulos en Rating:")
print(dataset_global["Rating"].isna().sum())


# =========================
# Guardar dataset
# =========================

dataset_global.to_csv(
    "datasets/dataset_unico.csv",
    index=False
)

print("\nDataset guardado correctamente.")