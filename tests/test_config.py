import pytest

from bicis import config


def test_rutas_relativas_se_resuelven_desde_params(entorno):
    assert config.ruta("fuente_db") == entorno.raiz / "fuente" / "bicis.db"
    assert config.ruta("modelo") == entorno.modelo


def test_variable_de_entorno_tiene_prioridad(entorno, monkeypatch, tmp_path):
    monkeypatch.setenv("BICIS_SALIDAS_DB", str(tmp_path / "otra.db"))
    assert config.ruta("salidas_db") == tmp_path / "otra.db"


@pytest.mark.parametrize("fecha", ["2011-13-01", "2011-1-5", "ayer", "", "2011-01-01T00:00"])
def test_fecha_invalida(fecha):
    with pytest.raises(ValueError, match="Fecha inválida"):
        config.validar_fecha(fecha)


def test_features_rechaza_columnas_prohibidas(entorno):
    import yaml
    p = yaml.safe_load(entorno.params.read_text())
    p["modelo"]["features"].append("registered")
    entorno.params.write_text(yaml.safe_dump(p))
    with pytest.raises(ValueError, match="registered"):
        config.features()


def test_features_no_incluye_casual_registered_ni_cnt(entorno):
    assert not {"casual", "registered", "cnt"} & set(config.features())
