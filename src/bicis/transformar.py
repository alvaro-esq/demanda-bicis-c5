"""Paso 3: prepara las features que usará el modelo.

Deja dteday, hr y las columnas de modelo.features (params.yaml), en ese orden.
Descarta todo lo demás (por ejemplo hora_imputada) y falla si aparece una
columna prohibida (casual, registered, cnt, ...).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bicis import config

ENTRADA = "limpio.csv"
SALIDA = "features.csv"


def columnas_salida() -> list[str]:
    return list(dict.fromkeys(["dteday", "hr", *config.features()]))


def ejecutar(fecha) -> Path:
    fecha = config.validar_fecha(fecha)
    df = pd.read_csv(config.leer_paso_anterior(fecha, ENTRADA))

    prohibidas = (config.prohibidas() - {"dteday"} | {"cnt"}).intersection(df.columns)
    if prohibidas:
        raise ValueError(f"Columnas prohibidas en {ENTRADA}: {sorted(prohibidas)}")
    cols = columnas_salida()
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en {ENTRADA}: {faltan}")

    salida = config.salida_paso(fecha, SALIDA)
    df[cols].sort_values("hr").to_csv(salida, index=False)
    return salida
