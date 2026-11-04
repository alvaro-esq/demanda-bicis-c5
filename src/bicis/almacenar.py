"""Paso 5: guarda las predicciones del día en data/salidas/predicciones.db.

Reemplaza el día completo: en una sola transacción borra todas las filas de
la fecha e inserta las nuevas. Repetir un día no duplica filas. Cada fila guarda
version_modelo, la versión del modelo registrado que hizo la predicción.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bicis import config

ENTRADA = "predicciones.csv"
TABLA = "predicciones"

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLA} (
    fecha TEXT NOT NULL,
    hr INTEGER NOT NULL CHECK (hr BETWEEN 0 AND 23),
    cnt_predicho INTEGER NOT NULL CHECK (cnt_predicho >= 0),
    creado_en TEXT NOT NULL,
    version_modelo TEXT,
    PRIMARY KEY (fecha, hr)
)
"""


def _migrar(con: sqlite3.Connection) -> None:
    """Bases creadas antes de version_modelo: se agrega la columna (filas viejas quedan NULL)."""
    columnas = {fila[1] for fila in con.execute(f"PRAGMA table_info({TABLA})")}
    if "version_modelo" not in columnas:
        con.execute(f"ALTER TABLE {TABLA} ADD COLUMN version_modelo TEXT")


def ejecutar(fecha) -> Path:
    fecha = config.validar_fecha(fecha)
    df = pd.read_csv(config.leer_paso_anterior(fecha, ENTRADA))

    faltan = {"fecha", "hr", "cnt_predicho", "version_modelo"} - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en {ENTRADA}: {sorted(faltan)}")
    if df.empty:
        raise ValueError(f"{ENTRADA} no tiene filas")
    if (df["fecha"].astype(str) != fecha).any():
        raise ValueError(f"{ENTRADA} tiene filas de otra fecha distinta de {fecha}")
    if df["hr"].duplicated().any():
        raise ValueError(f"{ENTRADA} tiene horas repetidas")

    creado_en = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if df["version_modelo"].isna().any():
        raise ValueError(f"{ENTRADA} tiene filas sin version_modelo")
    filas = [(fecha, int(h), int(c), creado_en, str(v))
             for h, c, v in zip(df["hr"], df["cnt_predicho"], df["version_modelo"])]

    db = config.ruta("salidas_db")
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    try:
        with con:  # una transacción: o se reemplaza el día entero o nada
            con.execute(DDL)
            _migrar(con)
            con.execute(f"DELETE FROM {TABLA} WHERE fecha = ?", (fecha,))
            con.executemany(
                f"INSERT INTO {TABLA} (fecha, hr, cnt_predicho, creado_en, version_modelo) "
                "VALUES (?, ?, ?, ?, ?)",
                filas,
            )
    finally:
        con.close()
    return db
