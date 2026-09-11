
LIGAS = [
    "Bundesliga",
    "La Liga",
    "Serie A",
    "Ligue 1",
    "Premier League"
]

PREFIJO_SOFASCORE_POR_LIGA = {
    "Bundesliga": "Bundesliga",
    "La Liga": "laliga",
    "Serie A": "SerieA",
    "Ligue 1": "ligue1",
    "Premier League": "PL"
}

ANIOS_DISPONIBLES = [2017, 2018, 2019, 2020, 2021, 2022, 2023]


def construir_temporadas():

    temporadas = {}

    for anio in ANIOS_DISPONIBLES:

        anio_siguiente = anio + 1
        nombre_temporada = f"{anio}-{str(anio_siguiente)[-2:]}"
        carpeta_sofascore = f"{anio}-{anio_siguiente}"
        sufijo_sofascore = f"{anio}{anio_siguiente}"

        temporadas[nombre_temporada] = {
            "fbref": f"datasets/FBref/cleaned_{nombre_temporada}.csv",

            "sofascore": {
                liga: (
                    f"datasets/sofascore/{carpeta_sofascore}/"
                    f"{prefijo}{sufijo_sofascore}.csv"
                )
                for liga, prefijo in PREFIJO_SOFASCORE_POR_LIGA.items()
            }
        }

    return temporadas


TEMPORADAS = construir_temporadas()
