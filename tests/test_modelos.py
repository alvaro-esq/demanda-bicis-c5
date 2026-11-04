import numpy as np
import pandas as pd
import pytest

from bicis import config, entrenar, modelos


def test_perfil_historico_promedia_por_hora_y_tipo_de_dia():
    X = pd.DataFrame({"hr": [8, 8, 8, 9], "workingday": [1, 1, 0, 1], "temp": [0.1, 0.9, 0.5, 0.5]})
    m = modelos.perfil_historico().fit(X, [100, 200, 30, 50])
    nuevo = pd.DataFrame({"hr": [8, 8, 9, 23], "workingday": [1, 0, 1, 1], "temp": [0, 0, 0, 0]})
    assert m.predict(nuevo).tolist() == [150, 30, 50, 95]   # 23 no visto -> media global
    assert list(m.feature_names_in_) == ["hr", "workingday", "temp"]


def test_ridge_usa_hora_y_mes_como_categorias(entorno):
    m = modelos.ridge(features=config.features())
    prep = m.named_steps["prep"]
    cats = dict((n, c) for n, _, c in prep.transformers)
    assert cats["categoricas"] == ["hr", "mnth"]
    assert "hr" not in cats["numericas"] and "mnth" not in cats["numericas"]


def test_hgb_entrena_con_el_objetivo_en_logaritmo():
    m = modelos.hist_gradient_boosting(max_iter=5)
    assert m.func is np.log1p and m.inverse_func is np.expm1


@pytest.mark.parametrize("tipo", ["perfil_historico", "ridge", "hist_gradient_boosting"])
def test_los_candidatos_entrenan_sin_tocar_produccion(entorno, tipo):
    m = entrenar.entrenar(tipo, registrar=False, guardar=False)
    assert m["modelo_tipo"] == tipo and m["mae_validacion"] > 0
    assert 0 <= m["factor_ajuste"] <= 0.40
    assert not entorno.modelo.exists() and not entorno.metricas.exists()


def test_perfil_historico_coincide_con_la_referencia(entorno):
    m = entrenar.entrenar("perfil_historico", registrar=False, guardar=False)
    # La referencia es el mismo promedio, sin redondear: difieren menos de 0.5 por hora.
    assert abs(m["mae_validacion"] - m["mae_referencia_promedio_hora"]) < 0.5
