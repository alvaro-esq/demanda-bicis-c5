import hashlib

import pandas as pd
import pytest

from bicis import extraer


def test_extrae_las_filas_del_dia(entorno):
    ruta = extraer.ejecutar("2011-06-15")
    assert ruta == entorno.staging / "2011-06-15" / "extraido.csv"
    df = pd.read_csv(ruta)
    assert len(df) == 24
    assert (df["dteday"] == "2011-06-15").all()
    assert df["hr"].tolist() == list(range(24))


def test_no_extrae_objetivo_ni_componentes(entorno):
    df = pd.read_csv(extraer.ejecutar("2011-06-15"))
    assert not {"cnt", "casual", "registered", "instant", "yr"} & set(df.columns)


def test_dia_sin_filas_escribe_archivo_vacio(entorno):
    df = pd.read_csv(extraer.ejecutar("2015-01-01"))
    assert df.empty and "hr" in df.columns


def test_no_modifica_la_fuente(entorno):
    antes = hashlib.md5(entorno.fuente_db.read_bytes()).hexdigest()
    extraer.ejecutar("2011-06-15")
    assert hashlib.md5(entorno.fuente_db.read_bytes()).hexdigest() == antes


def test_falla_si_no_existe_la_fuente(entorno):
    entorno.fuente_db.unlink()
    with pytest.raises(FileNotFoundError):
        extraer.ejecutar("2011-06-15")


def test_es_idempotente(entorno):
    a = extraer.ejecutar("2011-06-15").read_text()
    assert extraer.ejecutar("2011-06-15").read_text() == a
