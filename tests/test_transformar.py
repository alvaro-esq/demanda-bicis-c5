import pandas as pd
import pytest

from bicis import config, extraer, limpiar, transformar
from conftest import escribir_staging

FECHA = "2011-03-05"


def _limpio(entorno):
    extraer.ejecutar(FECHA)
    return pd.read_csv(limpiar.ejecutar(FECHA))


def test_deja_solo_claves_y_features_en_orden(entorno):
    _limpio(entorno)
    df = pd.read_csv(transformar.ejecutar(FECHA))
    otras = [f for f in config.features() if f != "hr"]
    assert list(df.columns) == ["dteday", "hr", *otras]
    assert len(df) == 24 and df["hr"].tolist() == list(range(24))
    assert "hora_imputada" not in df.columns


@pytest.mark.parametrize("col", ["casual", "registered", "cnt", "instant"])
def test_falla_con_columna_prohibida(entorno, col):
    escribir_staging(entorno, FECHA, "limpio.csv", _limpio(entorno).assign(**{col: 1}))
    with pytest.raises(ValueError, match="prohibidas"):
        transformar.ejecutar(FECHA)


def test_falla_si_falta_una_feature(entorno):
    escribir_staging(entorno, FECHA, "limpio.csv", _limpio(entorno).drop(columns="atemp"))
    with pytest.raises(ValueError, match="atemp"):
        transformar.ejecutar(FECHA)


def test_falla_si_falta_el_archivo_anterior(entorno):
    with pytest.raises(FileNotFoundError):
        transformar.ejecutar(FECHA)
