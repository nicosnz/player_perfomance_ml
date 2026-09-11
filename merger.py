
import unicodedata
import difflib
import pandas as pd


UMBRAL_SIMILITUD_NOMBRE = 0.75


ALIAS_CRUDOS = {
    "Alavés": "alaves", "Deportivo Alavés": "alaves",
    "Amiens": "amiens", "Amiens SC": "amiens",
    "Arminia": "arminia", "Arminia Bielefeld": "arminia",
    "Bayern Munich": "bayern", "FC Bayern München": "bayern",
    "Bochum": "bochum", "VfL Bochum 1848": "bochum",
    "Brest": "brest", "Stade Brestois": "brest",
    "Brighton": "brighton", "Brighton & Hove Albion": "brighton",
    "Chievo": "chievo", "ChievoVerona": "chievo",
    "Dortmund": "dortmund", "Borussia Dortmund": "dortmund",
    "Düsseldorf": "dusseldorf", "Fortuna Düsseldorf": "dusseldorf",
    "Eint Frankfurt": "frankfurt", "Eintracht Frankfurt": "frankfurt",
    "Freiburg": "freiburg", "SC Freiburg": "freiburg",
    "Gladbach": "gladbach", "Borussia M'gladbach": "gladbach",
    "Greuther Fürth": "furth", "SpVgg Greuther Fürth": "furth",
    "Heidenheim": "heidenheim", "1. FC Heidenheim": "heidenheim",
    "Hoffenheim": "hoffenheim", "TSG Hoffenheim": "hoffenheim",
    "Huddersfield": "huddersfield", "Huddersfield Town": "huddersfield",
    "Köln": "koln", "1. FC Köln": "koln",
    "La Coruña": "coruna", "Deportivo La Coruña": "coruna",
    "Lens": "lens", "RC Lens": "lens",
    "Levante": "levante", "Levante UD": "levante",
    "Leverkusen": "leverkusen", "Bayer 04 Leverkusen": "leverkusen",
    "Lyon": "lyon", "Olympique Lyonnais": "lyon",
    "Mainz 05": "mainz", "1. FSV Mainz 05": "mainz",
    "Manchester Utd": "man utd", "Manchester United": "man utd",
    "Marseille": "marseille", "Olympique de Marseille": "marseille",
    "Monaco": "monaco", "AS Monaco": "monaco",
    "Newcastle Utd": "newcastle", "Newcastle United": "newcastle",
    "Nott'ham Forest": "nottingham", "Nottingham Forest": "nottingham",
    "Nîmes": "nimes", "Nîmes Olympique": "nimes",
    "Nürnberg": "nurnberg", "1. FC Nürnberg": "nurnberg",
    "Paderborn 07": "paderborn", "SC Paderborn 07": "paderborn",
    "Paris S-G": "psg", "Paris Saint-Germain": "psg",
    "Reims": "reims", "Stade de Reims": "reims",
    "Rennes": "rennes", "Stade Rennais": "rennes",
    "Sampdoria": "sampdoria", "U.C Sampdoria": "sampdoria",
    "Sheffield Utd": "sheffield utd", "Sheffield United": "sheffield utd",
    "Strasbourg": "strasbourg", "RC Strasbourg": "strasbourg",
    "Stuttgart": "stuttgart", "VfB Stuttgart": "stuttgart",
    "Tottenham": "tottenham", "Tottenham Hotspur": "tottenham",
    "Union Berlin": "union berlin", "1. FC Union Berlin": "union berlin",
    "Werder Bremen": "werder bremen", "SV Werder Bremen": "werder bremen",
    "West Brom": "west brom", "West Bromwich Albion": "west brom",
    "West Ham": "west ham", "West Ham United": "west ham",
    "Wolfsburg": "wolfsburg", "VfL Wolfsburg": "wolfsburg",
    "Wolves": "wolves", "Wolverhampton": "wolves"
}


def limpiar_texto(texto):

    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))

    return texto.lower().strip()


ALIAS_EQUIPOS = {
    limpiar_texto(crudo): canonico
    for crudo, canonico in ALIAS_CRUDOS.items()
}


def normalizar_equipo(nombre):

    limpio = limpiar_texto(nombre)

    return ALIAS_EQUIPOS.get(limpio, limpio)


def tokens(nombre):

    texto = limpiar_texto(nombre).replace("-", " ")

    return set(texto.split())


def tokens_ordenados(nombre):
    return " ".join(sorted(tokens(nombre)))


def detectar_nombres_ambiguos(fbref_liga):

    personas_por_nombre = fbref_liga.groupby("player")["born"].nunique()

    return set(
        personas_por_nombre[personas_por_nombre > 1].index
    )


def emparejar_por_similitud(fbref_restante, sofascore_restante):

    emparejamientos = []
    sofascore_usado = set()

    for equipo, grupo_fbref in fbref_restante.groupby("equipo_norm"):

        candidatos = sofascore_restante[
            sofascore_restante["equipo_norm"] == equipo
        ]

        if candidatos.empty:
            continue

        for idx_fb, fila_fb in grupo_fbref.iterrows():

            nombre_fb = limpiar_texto(fila_fb["player"])
            tokens_fb = tokens(fila_fb["player"])

            mejor_idx = None
            mejor_score = 0.0

            for idx_so, fila_so in candidatos.iterrows():

                if idx_so in sofascore_usado:
                    continue

                nombre_so = limpiar_texto(fila_so["Player Name"])
                tokens_so = tokens(fila_so["Player Name"])

                if tokens_fb == tokens_so:
                    mejor_idx = idx_so
                    mejor_score = 1.0
                    break

                if tokens_fb and tokens_so and (
                    tokens_fb.issubset(tokens_so) or tokens_so.issubset(tokens_fb)
                ):
                    mejor_idx = idx_so
                    mejor_score = 0.95
                    break

                score = difflib.SequenceMatcher(None, nombre_fb, nombre_so).ratio()

                if score > mejor_score:
                    mejor_score = score
                    mejor_idx = idx_so

            if mejor_idx is not None and mejor_score >= UMBRAL_SIMILITUD_NOMBRE:
                emparejamientos.append((idx_fb, mejor_idx))
                sofascore_usado.add(mejor_idx)

    return emparejamientos


def crear_dataset_liga(fbref, liga, ruta_sofascore):

    fbref_liga = fbref[fbref["comp"] == liga].copy()
    fbref_liga["_idx_fbref"] = fbref_liga.index
    fbref_liga["equipo_norm"] = fbref_liga["squad"].apply(normalizar_equipo)

    sofascore = pd.read_csv(ruta_sofascore)
    sofascore_rating = sofascore[["Player Name", "Team Name", "Rating"]].copy()
    sofascore_rating["_idx_sofa"] = sofascore_rating.index
    sofascore_rating["equipo_norm"] = sofascore_rating["Team Name"].apply(normalizar_equipo)

    # =========================
    # Nivel 1: nombre exacto (sin ambigüedad de identidad)
    # =========================

    nombres_ambiguos = detectar_nombres_ambiguos(fbref_liga)
    es_ambiguo = fbref_liga["player"].isin(nombres_ambiguos)

    dataset_normal = fbref_liga[~es_ambiguo].merge(
        sofascore_rating[["Player Name", "Rating", "_idx_sofa"]],
        left_on="player",
        right_on="Player Name",
        how="inner"
    )

    # =========================
    # Nivel 2: nombre ambiguo -> exigir también el equipo
    # =========================

    dataset_ambiguo = fbref_liga[es_ambiguo].merge(
        sofascore_rating[["Player Name", "Rating", "equipo_norm", "_idx_sofa"]],
        left_on=["player", "equipo_norm"],
        right_on=["Player Name", "equipo_norm"],
        how="inner"
    )

    emparejado_exacto = pd.concat(
        [dataset_normal, dataset_ambiguo],
        ignore_index=True
    )

    # =========================
    # Nivel 3: nombres parecidos pero no idénticos
    # (orden distinto, tildes, nombre incompleto), restringido
    # al mismo equipo para no mezclar jugadores distintos
    # =========================

    idx_fbref_usados = set(emparejado_exacto["_idx_fbref"])
    idx_sofa_usados = set(emparejado_exacto["_idx_sofa"])

    fbref_restante = fbref_liga[~fbref_liga["_idx_fbref"].isin(idx_fbref_usados)]
    sofascore_restante = sofascore_rating[~sofascore_rating["_idx_sofa"].isin(idx_sofa_usados)]

    emparejamientos_similitud = emparejar_por_similitud(
        fbref_restante,
        sofascore_restante
    )

    filas_similitud = []

    for idx_fb, idx_so in emparejamientos_similitud:
        fila = fbref_restante.loc[idx_fb].to_dict()
        fila["Rating"] = sofascore_restante.loc[idx_so, "Rating"]
        filas_similitud.append(fila)

    dataset_similitud = pd.DataFrame(filas_similitud)

    dataset = pd.concat(
        [emparejado_exacto, dataset_similitud],
        ignore_index=True
    )

    columnas_auxiliares = [
        "Player Name", "equipo_norm", "_idx_fbref", "_idx_sofa"
    ]

    dataset = dataset.drop(
        columns=[c for c in columnas_auxiliares if c in dataset.columns]
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
