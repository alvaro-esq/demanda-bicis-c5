"""Configuración compartida: rutas, parámetros y utilidades comunes.

Módulo auxiliar (no es un paso del pipeline). Cada ruta sale de su variable de
entorno BICIS_<CLAVE>; si no está definida, se usa params.yaml. Las rutas
relativas se resuelven desde la carpeta que contiene params.yaml.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]


def ruta_params() -> Path:
    return Path(os.environ.get("BICIS_PARAMS", RAIZ / "params.yaml")).expanduser().resolve()


def params() -> dict:
    """Lee params.yaml en cada llamada, para que los cambios de entorno se respeten."""
    with open(ruta_params(), encoding="utf-8") as f:
        return yaml.safe_load(f)


def ruta(clave: str) -> Path:
    """Ruta de `rutas.<clave>`: primero BICIS_<CLAVE>, luego params.yaml."""
    valor = os.environ.get(f"BICIS_{clave.upper()}")
    if valor is None:
        try:
            valor = params()["rutas"][clave]
        except KeyError:
            raise KeyError(f"rutas.{clave} no está en {ruta_params()}") from None
    p = Path(valor).expanduser()
    return p if p.is_absolute() else ruta_params().parent / p


def validar_fecha(fecha: str | date) -> str:
    """Devuelve la fecha como 'AAAA-MM-DD' o lanza ValueError."""
    if isinstance(fecha, datetime):
        fecha = fecha.date()
    if isinstance(fecha, date):
        return fecha.isoformat()
    texto = str(fecha)
    if len(texto) == 10:
        try:
            return date.fromisoformat(texto).isoformat()
        except ValueError:
            pass
    raise ValueError(f"Fecha inválida {fecha!r}: se espera 'AAAA-MM-DD'")


def dir_staging(fecha: str) -> Path:
    return ruta("staging_dir") / fecha


def tabla_fuente() -> str:
    return params()["fuente"]["tabla"]


def features() -> list[str]:
    """Features del modelo. Falla si params.yaml incluye una columna prohibida."""
    m = params()["modelo"]
    feats = list(m["features"])
    prohibidas = set(m["prohibidas"]) | {"casual", "registered"}
    malas = prohibidas.intersection(feats) | ({m["objetivo"]} & set(feats))
    if malas:
        raise ValueError(f"params.yaml usa columnas prohibidas como features: {sorted(malas)}")
    return feats


def prohibidas() -> set[str]:
    return set(params()["modelo"]["prohibidas"]) | {"casual", "registered"}


def conectar_fuente() -> sqlite3.Connection:
    """Conexión de solo lectura a la fuente: el pipeline nunca la escribe."""
    db = ruta("fuente_db")
    if not db.exists():
        raise FileNotFoundError(f"No existe la fuente {db}. Corre `make fuente`.")
    return sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)


def leer_paso_anterior(fecha: str, nombre: str) -> Path:
    archivo = dir_staging(fecha) / nombre
    if not archivo.exists():
        raise FileNotFoundError(f"Falta {archivo}: corre antes el paso anterior")
    return archivo


def salida_paso(fecha: str, nombre: str) -> Path:
    d = dir_staging(fecha)
    d.mkdir(parents=True, exist_ok=True)
    return d / nombre
