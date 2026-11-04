"""Paso 1: toma de la fuente las filas del día.

Solo extrae lo que se conoce por la mañana: dteday, hr y las features del
modelo. Nunca lee cnt, casual ni registered. Si el día no tiene filas, escribe
un archivo vacío (solo encabezado); el que falla es limpiar.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bicis import config

SALIDA = "extraido.csv"


def columnas() -> list[str]:
    cols = list(dict.fromkeys(["dteday", "hr", *config.features()]))
    # dteday está prohibida como feature, pero aquí solo es la clave del día.
    if (config.prohibidas() - {"dteday"} | {"cnt"}).intersection(cols):
        raise ValueError(f"extraer no debe leer columnas prohibidas: {cols}")
    return cols


def ejecutar(fecha) -> Path:
    fecha = config.validar_fecha(fecha)
    cols = columnas()
    consulta = (
        f"SELECT {', '.join(cols)} FROM {config.tabla_fuente()} "
        "WHERE dteday = ? ORDER BY hr"
    )
    con = config.conectar_fuente()
    try:
        df = pd.read_sql_query(consulta, con, params=(fecha,))
    finally:
        con.close()

    salida = config.salida_paso(fecha, SALIDA)
    df.to_csv(salida, index=False)
    return salida
