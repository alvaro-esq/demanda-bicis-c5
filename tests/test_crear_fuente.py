import importlib.util
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("crear_fuente", RAIZ / "scripts" / "crear_fuente.py")
crear_fuente = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crear_fuente)


def _fila(instant, dteday, hr, yr, cnt):
    return {
        "instant": instant, "dteday": dteday, "season": 1, "yr": yr, "mnth": 1, "hr": hr,
        "holiday": 0, "weekday": 6, "workingday": 0, "weathersit": 1, "temp": 0.24,
        "atemp": 0.2879, "hum": 0.81, "windspeed": 0.0, "casual": 3,
        "registered": cnt - 3, "cnt": cnt,
    }


def _zip_con(tmp_path: Path, filas: list[dict]) -> Path:
    ruta = tmp_path / "dataset.zip"
    with zipfile.ZipFile(ruta, "w") as zf:
        zf.writestr("Readme.txt", "x")
        zf.writestr("hour.csv", pd.DataFrame(filas).to_csv(index=False))
    return ruta


def test_crea_tabla_con_todas_las_filas(tmp_path):
    filas = [
        _fila(1, "2011-01-01", 0, 0, 16),
        _fila(2, "2011-01-01", 1, 0, 40),
        _fila(3, "2012-01-01", 0, 1, 48),
    ]
    db = tmp_path / "fuente" / "bicis.db"

    assert crear_fuente.main(["--zip", str(_zip_con(tmp_path, filas)), "--db", str(db)]) == 0

    with sqlite3.connect(db) as con:
        df = pd.read_sql("SELECT * FROM registros_horarios ORDER BY instant", con)
    assert list(df.columns) == list(crear_fuente.COLUMNAS)
    assert df["cnt"].tolist() == [16, 40, 48]
    assert crear_fuente.filas_por_anio(db, "registros_horarios") == [("2011", 2), ("2012", 1)]


def test_reejecutar_reemplaza_la_base(tmp_path):
    db = tmp_path / "bicis.db"
    zip1 = _zip_con(tmp_path, [_fila(1, "2011-01-01", 0, 0, 16), _fila(2, "2011-01-01", 1, 0, 40)])
    crear_fuente.main(["--zip", str(zip1), "--db", str(db)])
    zip2 = _zip_con(tmp_path, [_fila(1, "2011-01-01", 0, 0, 16)])
    crear_fuente.main(["--zip", str(zip2), "--db", str(db)])

    assert crear_fuente.filas_por_anio(db, "registros_horarios") == [("2011", 1)]
    assert not list(tmp_path.glob("*.tmp"))


def test_rechaza_horas_duplicadas(tmp_path):
    filas = [_fila(1, "2011-01-01", 0, 0, 16), _fila(2, "2011-01-01", 0, 0, 40)]
    with pytest.raises(ValueError, match="duplicados"):
        crear_fuente.leer_hour_csv(_zip_con(tmp_path, filas))
