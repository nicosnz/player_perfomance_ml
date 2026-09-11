<div align="center">

# Player Performance ML

![Python](https://img.shields.io/badge/Python-3.12-blue)
![pandas](https://img.shields.io/badge/pandas-2.3-150458)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E)
![Modelo](https://img.shields.io/badge/Modelo-Regresi%C3%B3n%20Lineal-success)
![Estado](https://img.shields.io/badge/Estado-En%20desarrollo-yellow)

Predictor del rating de un jugador de fútbol para la temporada siguiente, a partir de sus estadísticas de la temporada actual.

</div>

Es un proyecto de **regresión lineal** (sin modelos no lineales), que combina estadísticas de juego (FBref) con el rating de rendimiento (SofaScore) de las 5 grandes ligas europeas, a lo largo de 7 temporadas.

<div align="center">

| | |
|---|---|
| **Temporadas** | 2017-18 a 2023-24 (7) |
| **Ligas** | Bundesliga · La Liga · Serie A · Ligue 1 · Premier League |
| **Dataset de entrenamiento** | 7244 filas · 53 features |
| **Match rate FBref ↔ SofaScore** | ~72-75% |
| **Mejor modelo** | Lasso / ElasticNet |
| **R² (walk-forward)** | 0.396 vs 0.258 del baseline |

</div>

---

## Índice

- [Fuentes de datos](#fuentes-de-datos)
- [1. Recolección de datos](#1-recolección-de-datos)
- [2. El merge (FBref + SofaScore)](#2-el-merge-fbref--sofascore)
- [3. Identidad de jugador y construcción del target](#3-identidad-de-jugador-y-construcción-del-target)
- [4. Limpieza y features finales](#4-limpieza-y-features-finales)
- [5. Correlación con el target](#5-correlación-con-el-target)
- [6. Entrenamiento del modelo](#6-entrenamiento-del-modelo)
- [7. ¿La regresión lineal es el modelo adecuado?](#7-la-regresión-lineal-es-el-modelo-adecuado)
- [8. Limitaciones conocidas](#8-limitaciones-conocidas)
- [Estructura del pipeline](#estructura-del-pipeline)

---

## Fuentes de datos

- FBref (stats de juego, 2017-2024, top 5 ligas europeas): https://www.kaggle.com/datasets/akshankrithick/fbref-2017-2024-for-europes-top-5-leagues?select=cleaned_2022-23.csv
- SofaScore (rating por jugador/temporada): https://www.kaggle.com/datasets/akshankrithick/sofascore-seasonwise-ratings-football-soccer?utm_source=chatgpt.com&select=ligue120222023.csv

---

## 1. Recolección de datos

Dos fuentes de Kaggle, jugadores de las 5 grandes ligas europeas (Bundesliga, La Liga, Serie A, Ligue 1, Premier League), **7 temporadas** (2017-18 a 2023-24):

- **FBref**: stats de juego (goles, pases, defensa, etc.) — 65 columnas por jugador/temporada, ~2400-2600 jugadores por temporada.
- **SofaScore**: el rating de rendimiento (target), un archivo por liga y por temporada.

---

## 2. El merge (FBref + SofaScore)

Unir las dos fuentes por nombre de jugador no es trivial porque cada una escribe los nombres distinto. El matching se hace en 3 niveles, por liga y por temporada:

1. **Nombre exacto** — cubre la mayoría de los casos.
2. **Nombre ambiguo**: mismo nombre pero dos personas reales distintas (encontramos 41 casos reales, ej. dos jugadores llamados "Aaron Ramsey" con distinto año de nacimiento). Para estos casos se exige que además coincida el equipo, usando un diccionario de 46 alias de club armado a mano porque tampoco coinciden los nombres de equipo entre fuentes (`"Wolves"` vs `"Wolverhampton"`, `"Dortmund"` vs `"Borussia Dortmund"`, `"Paris S-G"` vs `"Paris Saint-Germain"`, etc.).
3. **Nombre parecido pero no idéntico** (orden invertido, nombre incompleto — ej. `"Son Heung-min"` vs `"Heung-min Son"`, `"Pierre Højbjerg"` vs `"Pierre-Emile Højbjerg"`): similitud de texto, siempre restringida al mismo equipo para no arriesgar falsos positivos.

Esto llevó el match rate de ~66% (con menos temporadas y matching más simple) a **~72-75%**. Lo que queda sin matchear es mayoritariamente jugadores con pocos minutos que SofaScore directamente no trackea (verificado: los no-matcheados juegan en promedio 338 min por temporada vs 1870 min de los que sí matchean).

---

## 3. Identidad de jugador y construcción del target

Como dos personas distintas pueden compartir nombre, se usa `(player, born)` como identificador en vez de solo el nombre, para no mezclar historias de carrera de gente distinta.

El target (`rating_siguiente_temporada`) se arma buscando, para cada fila, el rating de ese mismo jugador **en la temporada siguiente**. Si no existe (se retiró, salió de las 5 ligas, o le faltó esa temporada puntual), el target queda `NaN` y esa fila se excluye del entrenamiento — no se inventa ningún valor.

Las transferencias a mitad de temporada generan 2 filas por jugador (una por club) con el mismo rating repetido en ambas — se dedupean quedándose con la primera.

---

## 4. Limpieza y features finales

- Se separan los **porteros** (posición GK) en un dataset aparte — sus métricas (`Saves`, `Goals Against`, etc.) son 0 para jugadores de campo, puro ruido si se mezclan. Quedan pendientes para un modelo aparte, todavía no entrenado.
- Se sacan columnas duplicadas exactas (`Goals Scored` = `Goals`, `carries_prgc` = `Progressive Carries`) e identificadores que no sirven como feature (`rk`, `born`, `nation`, `squad`, `comp`).
- `pos` se codifica one-hot (`pos_DF` / `pos_FW` / `pos_MF`).
- `season` se convierte en `anio_inicio` (numérico), en vez de descartarla o codificarla como categoría — así el modelo puede captar tendencia entre años y generaliza a temporadas futuras.

**Dataset final de entrenamiento**: `dataset_entrenamiento_campo.csv` — 7244 filas, 53 features numéricas + `player` de referencia.

---

## 5. Correlación con el target

| Feature                    | r     |
| -------------------------- | ----- |
| `Rating` (actual)          | 0.626 |
| `Key passes`               | 0.424 |
| `Goals & Assists`          | 0.414 |
| `Passes into penalty area` | 0.402 |
| `Progressive Passes`       | 0.384 |
| `Total Shots`              | 0.383 |
| `Assists`                  | 0.382 |
| `Goals`                    | 0.345 |
| `Passes Attempted`         | 0.338 |
| `Expected Goals`           | 0.331 |

El propio rating actual es, lejos, el mejor predictor individual — el resto son señales de producción ofensiva y creación de juego, todas con correlación moderada (r < 0.45). No hay una sola feature "mágica".

**Features más débiles** (|r| < 0.05, candidatas a sacar si se busca un modelo más chico): `anio_inicio`, `age`, `Shots blocked`, `% Aerial Duels won`, `pos_MF`, `% Dribbles tackled`, `Clearances`.

**Multicolinealidad** (grupos de features que dicen casi lo mismo entre sí, |r| ≥ 0.85, no resuelta del todo):

- Goles/xG: `Goals` ↔ `Non Penalty Goals` ↔ `Expected Goals` ↔ `Exp NPG` ↔ `Goals & Assists` ↔ `Total Shots` ↔ `Goals p 90`.
- Pases/posesión: `Passes Completed` ↔ `Passes Attempted` ↔ `Progressive passes distance` ↔ `1/3` ↔ `Progressive Passes`; y `Pass completion %` ↔ `% Short/Medium passes completed`.
- Defensa: `Tackles attempted` ↔ `Tackles Won`; `Clearances` ↔ `touches_def_pen` ↔ `Shots blocked`.
- Progresión con balón: `Progressive Carries` ↔ `carries final 3rd`.
- Minutos jugados: `Matches Played` ↔ `Avg Mins per Match`.
- Creación: `Key passes` ↔ `Passes into penalty area`.

---

## 6. Entrenamiento del modelo

**Restricción del proyecto**: solo regresión lineal (sin modelos no lineales). Se comparan 4 variantes:

- **OLS**: regresión lineal común, minimiza solo el error de predicción.
- **Ridge (L2)**: le suma una penalización proporcional a la suma de los coeficientes al cuadrado. Encoge todos los coeficientes hacia 0 pero nunca los deja en 0 exacto — reparte el peso entre features correlacionadas en vez de jugárselo todo a una.
- **Lasso (L1)**: penaliza la suma del valor absoluto de los coeficientes. A diferencia de Ridge, sí puede llevar coeficientes a 0 exacto — hace selección automática de features.
- **ElasticNet (L1+L2)**: combina ambas penalizaciones. Busca un punto intermedio: algo de selección de features como Lasso, pero sin ser tan agresivo eliminando features correlacionadas entre sí.

El `alpha` (qué tan fuerte es la penalización) no se elige a mano: `RidgeCV`/`LassoCV`/`ElasticNetCV` prueban muchos valores con cross-validation interna sobre el train de cada fold.

**Validación**: walk-forward. Para cada año disponible, se entrena con todos los años anteriores y se testea con ese año (5 folds: test 2018, 2019, 2020, 2021, 2022) — simula el caso real de predecir a futuro sin mezclar información que el modelo no tendría en producción.

**Estandarización**: `StandardScaler` ajustado solo con el train de cada fold, para no filtrar información del test.

**Feature engineering probado**: `age²` (curva de rendimiento por edad, no lineal) sí se mantuvo. Se probaron además una interacción `Rating × anio_inicio` y transformación logarítmica de `Goals`/`Assists` — ninguna de las dos mejoró el resultado de forma consistente (quedan descartadas, documentado el motivo: la interacción cruda queda colineal con sus propios componentes, y la regularización ya es bastante tolerante a la cola larga de los goles).

### Resultados (promedio de los 5 folds walk-forward)

| Modelo                              | MAE        | RMSE       | R²         |
| ----------------------------------- | ---------- | ---------- | ---------- |
| Baseline (repetir el rating actual) | 0.1457     | 0.1874     | 0.2580     |
| Baseline (promedio de train)        | 0.1684     | 0.2203     | -0.0232    |
| Regresión lineal (OLS)              | 0.1314     | 0.1699     | 0.3898     |
| Ridge (L2)                          | 0.1311     | 0.1694     | 0.3929     |
| **Lasso (L1)**                      | **0.1307** | **0.1690** | **0.3959** |
| ElasticNet (L1+L2)                  | 0.1307     | 0.1690     | 0.3959     |

Lasso y ElasticNet empatan como mejores, por muy poco margen sobre Ridge y OLS — con el volumen de datos actual (hasta 6017 filas de train en el último fold), la diferencia entre regularizar o no ya es chica.

---

## 7. ¿La regresión lineal es el modelo adecuado?

Evidencia a favor de que el techo actual no es la forma del modelo, sino los datos:

- OLS, Ridge, Lasso y ElasticNet dieron resultados casi idénticos (R² entre 0.388 y 0.396) — si hubiera una relación fuertemente no lineal sin capturar, se esperaría más diferencia entre ellos.
- Agregar términos no lineales a mano (interacción, log) no mejoró nada de forma consistente.
- Lo que sí movió la aguja fue tener más datos: pasar de 3 a 7 temporadas llevó el R² de ~0.16 a ~0.40.

Conclusión: la regresión lineal no parece estar dejando ganancia evidente sobre la mesa con los datos actuales. Es probable que un modelo no lineal (Random Forest, Gradient Boosting) mejore algo el resultado capturando interacciones entre features que acá habría que especificar a mano, pero la mejora esperable es moderada, no un salto grande — buena parte de la varianza no explicada es ruido inherente al fútbol (lesiones, cambios de DT, forma física) y pérdida de datos en el matching, no falta de flexibilidad del modelo.

---

## 8. Limitaciones conocidas

- **~25-28% de los jugadores de FBref siguen sin matchear** con SofaScore — sigue siendo la mejora de mayor impacto pendiente.
- **Escala de rating angosta** (la mayoría de los valores cae entre 6.5 y 7.2) — limita estructuralmente el R² alcanzable.
- **Multicolinealidad remanente** sin resolver manualmente — genera coeficientes inestables o con signo contraintuitivo en el modelo final (ej. `Avg Mins per Match` salió con coeficiente negativo pese a tener correlación simple positiva con el target).
- **`squad` y `nation` se descartaron enteras** como feature — probablemente hay señal de contexto de club perdida (jugar en un equipo grande vs uno chico).
- El fold de test 2022 rinde notablemente peor que los demás (R²≈0.19-0.21 vs 0.40-0.47 del resto) — anomalía detectada pero no investigada todavía.
- **Modelo de porteros**: dataset separado y listo, pero todavía no entrenado.
- Las transferencias a mitad de temporada se resuelven descartando una de las dos filas duplicadas, en vez de agregar las estadísticas de ambos clubes.

---

## Estructura del pipeline

```
config.py      -> rutas de los datasets por temporada/liga
loader.py      -> carga de CSVs
merger.py      -> matching FBref + SofaScore (3 niveles)
target.py      -> construcción del target (rating de la temporada siguiente)
validator.py   -> validación y limpieza de duplicados
exporter.py    -> guardado de datasets
main.py        -> orquesta todo lo anterior, genera dataset_historico / entrenamiento / prediccion
features.py    -> separa porteros, saca columnas basura/duplicadas, codifica features finales
correlacion.py -> matriz de correlación + detección de multicolinealidad
modelo.py      -> entrenamiento y evaluación walk-forward (OLS, Ridge, Lasso, ElasticNet)
```
