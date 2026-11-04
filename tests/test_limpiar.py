import pandas as pd
import pytest

from bicis import extraer, limpiar
from conftest import escribir_staging

FECHA = "2011-06-15"


def _extraido(entorno, fecha=FECHA):
    return pd.read_csv(extraer.ejecutar(fecha))


def test_dia_completo_queda_igual_y_sin_imputar(entorno):
    extraer.ejecutar(FECHA)
    df = pd.read_csv(limpiar.ejecutar(FECHA))
    assert len(df) == 24 and df["hr"].tolist() == list(range(24))
    assert (df["hora_imputada"] == 0).all()


def test_completa_horas_faltantes_con_la_vecina_mas_cercana(entorno):
    fecha = "2011-03-05"  # sin las horas 3, 4 y 23
    original = _extraido(entorno, fecha).set_index("hr")
    df = pd.read_csv(limpiar.ejecutar(fecha)).set_index("hr")
    assert df.index.tolist() == list(range(24))
    assert df.loc[[3, 4, 23], "hora_imputada"].tolist() == [1, 1, 1]
    assert df["hora_imputada"].sum() == 3
    assert df.loc[3, "temp"] == original.loc[2, "temp"]   # 3 está más cerca de 2 que de 5
    assert df.loc[4, "temp"] == original.loc[5, "temp"]   # 4 está más cerca de 5 que de 2
    assert df.loc[23, "temp"] == original.loc[22, "temp"]
    assert df.loc[3, "weekday"] == original.loc[0, "weekday"]


def test_falla_si_el_dia_no_tiene_filas(entorno):
    extraer.ejecutar("2015-01-01")
    with pytest.raises(ValueError, match="no tiene filas"):
        limpiar.ejecutar("2015-01-01")


def test_falla_si_falta_una_columna(entorno):
    escribir_staging(entorno, FECHA, "extraido.csv", _extraido(entorno).drop(columns="hum"))
    with pytest.raises(ValueError, match="Faltan columnas.*hum"):
        limpiar.ejecutar(FECHA)


@pytest.mark.parametrize("col", ["casual", "registered", "cnt"])
def test_falla_si_aparece_columna_prohibida(entorno, col):
    escribir_staging(entorno, FECHA, "extraido.csv", _extraido(entorno).assign(**{col: 1}))
    with pytest.raises(ValueError, match="prohibidas"):
        limpiar.ejecutar(FECHA)


@pytest.mark.parametrize("col,valor", [("hr", 24), ("weathersit", 5), ("temp", 1.5), ("hum", -0.1), ("mnth", 0)])
def test_falla_fuera_de_rango(entorno, col, valor):
    df = _extraido(entorno)
    df.loc[0, col] = valor
    escribir_staging(entorno, FECHA, "extraido.csv", df)
    with pytest.raises(ValueError, match=f"{col}.*fuera de"):
        limpiar.ejecutar(FECHA)


def test_falla_con_valores_no_numericos(entorno):
    df = _extraido(entorno).astype({"temp": object})
    df.loc[0, "temp"] = "caliente"
    escribir_staging(entorno, FECHA, "extraido.csv", df)
    with pytest.raises(ValueError, match="temp.*no numéricos"):
        limpiar.ejecutar(FECHA)


def test_falla_con_horas_repetidas(entorno):
    df = _extraido(entorno)
    df.loc[1, "hr"] = 0
    escribir_staging(entorno, FECHA, "extraido.csv", df)
    with pytest.raises(ValueError, match="repetidas"):
        limpiar.ejecutar(FECHA)


def test_falla_si_falta_el_archivo_anterior(entorno):
    with pytest.raises(FileNotFoundError):
        limpiar.ejecutar(FECHA)
