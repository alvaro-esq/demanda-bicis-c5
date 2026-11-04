"""Paso 2: revisa columnas, tipos y rangos, y deja exactamente 24 horas.

Falla (ValueError) si el día no tiene filas, si falta una columna, si aparece
una columna prohibida, si un valor no es numérico o está fuera de rango, o si
una hora está repetida. Las horas que faltan se agregan con el calendario del
día y el clima de la hora vecina más cercana, marcadas con hora_imputada = 1.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bicis import config

ENTRADA = "extraido.csv"
SALIDA = "limpio.csv"
HORAS = list(range(24))

# Columnas que describen el día: deben ser iguales en todas sus horas.
CALENDARIO = ["season", "mnth", "holiday", "weekday", "workingday"]
# Columnas que cambian hora a hora.
CLIMA = ["weathersit", "temp", "atemp", "hum", "windspeed"]
ENTERAS = {"hr", "season", "mnth", "holiday", "weekday", "workingday", "weathersit"}

RANGOS = {
    "hr": (0, 23),
    "season": (1, 4),
    "mnth": (1, 12),
    "holiday": (0, 1),
    "weekday": (0, 6),
    "workingday": (0, 1),
    "weathersit": (1, 4),
    "temp": (0.0, 1.0),
    "atemp": (0.0, 1.0),
    "hum": (0.0, 1.0),
    "windspeed": (0.0, 1.0),
}


def _validar(df: pd.DataFrame, fecha: str, cols: list[str]) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"El día {fecha} no tiene filas en la fuente")

    prohibidas = (config.prohibidas() - {"dteday"} | {"cnt"}).intersection(df.columns)
    if prohibidas:
        raise ValueError(f"Columnas prohibidas en {ENTRADA}: {sorted(prohibidas)}")
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en {ENTRADA}: {faltan}")
    df = df[cols].copy()

    if (df["dteday"].astype(str) != fecha).any():
        raise ValueError(f"{ENTRADA} tiene filas de otra fecha distinta de {fecha}")

    numericas = [c for c in cols if c != "dteday"]
    for c in numericas:
        valores = pd.to_numeric(df[c], errors="coerce")
        if valores.isna().any():
            raise ValueError(f"Columna {c}: valores vacíos o no numéricos")
        if c in ENTERAS:
            if (valores % 1 != 0).any():
                raise ValueError(f"Columna {c}: se esperaban enteros")
            valores = valores.astype("int64")
        else:
            valores = valores.astype("float64")
        if c in RANGOS:
            lo, hi = RANGOS[c]
            fuera = valores[(valores < lo) | (valores > hi)]
            if not fuera.empty:
                raise ValueError(f"Columna {c}: valores fuera de [{lo}, {hi}]: {fuera.tolist()[:5]}")
        df[c] = valores

    if df["hr"].duplicated().any():
        raise ValueError(f"Horas repetidas: {sorted(df.loc[df['hr'].duplicated(), 'hr'].unique())}")
    for c in [c for c in CALENDARIO if c in df.columns]:
        if df[c].nunique() > 1:
            raise ValueError(f"Columna {c}: debería ser igual en todas las horas del día")
    return df


def _completar_horas(df: pd.DataFrame, fecha: str) -> pd.DataFrame:
    presentes = df.set_index("hr").sort_index()
    filas = []
    for h in HORAS:
        if h in presentes.index:
            fila = presentes.loc[h].to_dict()
            fila["hora_imputada"] = 0
        else:
            # Hora vecina más cercana; en empate, la anterior.
            vecina = min(presentes.index, key=lambda p: (abs(p - h), p))
            fila = presentes.loc[vecina].to_dict()
            fila["hora_imputada"] = 1
        fila["hr"] = h
        filas.append(fila)
    completo = pd.DataFrame(filas)
    completo["dteday"] = fecha
    return completo


def ejecutar(fecha) -> Path:
    fecha = config.validar_fecha(fecha)
    entrada = config.leer_paso_anterior(fecha, ENTRADA)
    cols = list(dict.fromkeys(["dteday", "hr", *config.features()]))

    df = _validar(pd.read_csv(entrada), fecha, cols)
    df = _completar_horas(df, fecha)[[*cols, "hora_imputada"]]
    for c in cols[1:]:
        df[c] = df[c].astype("int64" if c in ENTERAS else "float64")

    salida = config.salida_paso(fecha, SALIDA)
    df.to_csv(salida, index=False)
    return salida
