"""Crea la base SQLite que simula el sistema de origen.

Lee hour.csv desde el zip del dataset Bike Sharing (UCI) y lo carga en la
tabla registros_horarios de fuente/bicis.db. El pipeline solo lee esta base;
este script es lo único que la escribe.

Uso:
    python scripts/crear_fuente.py [--zip RUTA] [--db RUTA]

Las rutas por defecto salen de params.yaml y se pueden sobrescribir con las
variables de entorno BICIS_ZIP_ORIGEN y BICIS_FUENTE_DB.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parent.parent

# Esquema del sistema de origen: mismas columnas y significado que hour.csv.
COLUMNAS = {
    "instant": "INTEGER NOT NULL",
    "dteday": "TEXT NOT NULL",       # fecha ISO, AAAA-MM-DD
    "season": "INTEGER NOT NULL",    # 1 invierno, 2 primavera, 3 verano, 4 otoño
    "yr": "INTEGER NOT NULL",        # 0 = 2011, 1 = 2012
    "mnth": "INTEGER NOT NULL",
    "hr": "INTEGER NOT NULL",        # 0..23
    "holiday": "INTEGER NOT NULL",
    "weekday": "INTEGER NOT NULL",
    "workingday": "INTEGER NOT NULL",
    "weathersit": "INTEGER NOT NULL",
    "temp": "REAL NOT NULL",         # normalizadas, 0..1
    "atemp": "REAL NOT NULL",
    "hum": "REAL NOT NULL",
    "windspeed": "REAL NOT NULL",
    "casual": "INTEGER NOT NULL",
    "registered": "INTEGER NOT NULL",
    "cnt": "INTEGER NOT NULL",
}


def cargar_params(ruta: Path = RAIZ / "params.yaml") -> dict:
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolver(valor: str | os.PathLike, base: Path = RAIZ) -> Path:
    ruta = Path(valor).expanduser()
    return ruta if ruta.is_absolute() else base / ruta


def leer_hour_csv(ruta_zip: Path, nombre_csv: str = "hour.csv") -> pd.DataFrame:
    with zipfile.ZipFile(ruta_zip) as zf:
        candidatos = [n for n in zf.namelist() if Path(n).name == nombre_csv]
        if not candidatos:
            raise FileNotFoundError(f"{nombre_csv} no está dentro de {ruta_zip}")
        with zf.open(candidatos[0]) as f:
            df = pd.read_csv(f)

    faltantes = set(COLUMNAS) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en {nombre_csv}: {sorted(faltantes)}")
    df = df[list(COLUMNAS)]
    # Normaliza la fecha a texto ISO para que el filtro por día sea exacto.
    df["dteday"] = pd.to_datetime(df["dteday"]).dt.strftime("%Y-%m-%d")
    if df.duplicated(["dteday", "hr"]).any():
        raise ValueError("Hay registros duplicados para el mismo (dteday, hr)")
    return df


def escribir_sqlite(df: pd.DataFrame, ruta_db: Path, tabla: str) -> None:
    """Escribe la base completa en un archivo temporal y lo reemplaza de forma atómica."""
    ruta_db.parent.mkdir(parents=True, exist_ok=True)
    columnas_sql = ",\n  ".join(f"{c} {t}" for c, t in COLUMNAS.items())
    ddl = (
        f"CREATE TABLE {tabla} (\n  {columnas_sql},\n"
        f"  PRIMARY KEY (dteday, hr)\n)"
    )

    fd, tmp = tempfile.mkstemp(dir=ruta_db.parent, suffix=".db.tmp")
    os.close(fd)
    try:
        with sqlite3.connect(tmp) as con:
            con.execute(ddl)
            con.execute(f"CREATE UNIQUE INDEX ux_{tabla}_instant ON {tabla}(instant)")
            marcadores = ", ".join("?" for _ in COLUMNAS)
            con.executemany(
                f"INSERT INTO {tabla} ({', '.join(COLUMNAS)}) VALUES ({marcadores})",
                df.itertuples(index=False, name=None),
            )
        con.close()
        os.replace(tmp, ruta_db)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def filas_por_anio(ruta_db: Path, tabla: str) -> list[tuple[str, int]]:
    with sqlite3.connect(f"file:{ruta_db}?mode=ro", uri=True) as con:
        return con.execute(
            f"SELECT substr(dteday, 1, 4) AS anio, COUNT(*) FROM {tabla} "
            "GROUP BY anio ORDER BY anio"
        ).fetchall()


def main(argv: list[str] | None = None) -> int:
    params = cargar_params()
    rutas, fuente = params["rutas"], params["fuente"]

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--zip",
        type=Path,
        default=resolver(os.environ.get("BICIS_ZIP_ORIGEN", rutas["zip_origen"])),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=resolver(os.environ.get("BICIS_FUENTE_DB", rutas["fuente_db"])),
    )
    args = parser.parse_args(argv)

    if not args.zip.exists():
        print(f"No encuentro el zip: {args.zip}", file=sys.stderr)
        return 1

    df = leer_hour_csv(args.zip, fuente["csv_en_zip"])
    escribir_sqlite(df, args.db, fuente["tabla"])

    print(f"{len(df)} filas -> {args.db} (tabla {fuente['tabla']})")
    for anio, n in filas_por_anio(args.db, fuente["tabla"]):
        print(f"  {anio}: {n} filas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
