import sqlite3

import pandas as pd
import pytest

from bicis import almacenar
from conftest import escribir_staging


def _predicciones(entorno, fecha, base=10):
    df = pd.DataFrame({"fecha": fecha, "hr": range(24), "cnt_predicho": [base + h for h in range(24)]})
    escribir_staging(entorno, fecha, "predicciones.csv", df)


def _leer(db):
    with sqlite3.connect(db) as con:
        return pd.read_sql("SELECT fecha, hr, cnt_predicho FROM predicciones ORDER BY fecha, hr", con)


def test_guarda_24_filas_y_devuelve_la_ruta(entorno):
    _predicciones(entorno, "2012-01-10")
    assert almacenar.ejecutar("2012-01-10") == entorno.salidas_db
    df = _leer(entorno.salidas_db)
    assert len(df) == 24 and df["cnt_predicho"].tolist() == [10 + h for h in range(24)]


def test_repetir_el_dia_no_duplica_y_reemplaza(entorno):
    _predicciones(entorno, "2012-01-10", base=10)
    almacenar.ejecutar("2012-01-10")
    almacenar.ejecutar("2012-01-10")
    _predicciones(entorno, "2012-01-10", base=100)
    almacenar.ejecutar("2012-01-10")
    df = _leer(entorno.salidas_db)
    assert len(df) == 24
    assert df["cnt_predicho"].tolist() == [100 + h for h in range(24)]


def test_no_toca_otros_dias(entorno):
    _predicciones(entorno, "2012-01-10", base=10)
    almacenar.ejecutar("2012-01-10")
    _predicciones(entorno, "2012-01-11", base=50)
    almacenar.ejecutar("2012-01-11")
    _predicciones(entorno, "2012-01-10", base=70)
    almacenar.ejecutar("2012-01-10")
    df = _leer(entorno.salidas_db)
    assert df.groupby("fecha").size().to_dict() == {"2012-01-10": 24, "2012-01-11": 24}
    assert df.loc[df["fecha"] == "2012-01-11", "cnt_predicho"].iloc[0] == 50


def test_reemplaza_el_dia_completo_aunque_cambie_el_numero_de_horas(entorno):
    _predicciones(entorno, "2012-01-10")
    almacenar.ejecutar("2012-01-10")
    df = pd.DataFrame({"fecha": "2012-01-10", "hr": [0, 1], "cnt_predicho": [5, 6]})
    escribir_staging(entorno, "2012-01-10", "predicciones.csv", df)
    almacenar.ejecutar("2012-01-10")
    assert _leer(entorno.salidas_db)["hr"].tolist() == [0, 1]


def test_si_falla_no_pierde_el_dia_anterior(entorno):
    _predicciones(entorno, "2012-01-10")
    almacenar.ejecutar("2012-01-10")
    df = pd.DataFrame({"fecha": "2012-01-10", "hr": [0, 1], "cnt_predicho": [5, -1]})  # viola CHECK
    escribir_staging(entorno, "2012-01-10", "predicciones.csv", df)
    with pytest.raises(sqlite3.IntegrityError):
        almacenar.ejecutar("2012-01-10")
    assert len(_leer(entorno.salidas_db)) == 24


def test_falla_si_el_archivo_es_de_otra_fecha(entorno):
    _predicciones(entorno, "2012-01-11")
    escribir_staging(entorno, "2012-01-10", "predicciones.csv",
                     pd.read_csv(entorno.staging / "2012-01-11" / "predicciones.csv"))
    with pytest.raises(ValueError, match="otra fecha"):
        almacenar.ejecutar("2012-01-10")
