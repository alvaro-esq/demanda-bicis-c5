import numpy as np
import pytest
import yaml

from bicis import config, costos

C = costos.Costos(faltante=3, sobrante=1)


# ---------------------------------------------------------- función de costo

@pytest.mark.parametrize("real,plan,esperado", [
    (10, 10, 0),     # plan exacto: sin costo
    (10, 8, 6),      # faltaron 2 -> 3 x 2
    (10, 12, 2),     # sobraron 2 -> 1 x 2
    (0, 5, 5),       # nadie alquiló: sobran todas
    (7, 0, 21),      # no se planificó nada: faltan todas
])
def test_costo_por_hora_caso_a_caso(real, plan, esperado):
    assert costos.costo_por_hora([real], [plan], C).tolist() == [esperado]


def test_costo_es_asimetrico_faltar_cuesta_el_triple():
    assert costos.costo_medio([10], [7], C) == 3 * costos.costo_medio([10], [13], C)


def test_costo_medio_por_hora_vectorizado():
    real = [10, 10, 10, 10]
    plan = [10, 8, 12, 11]   # 0 + 6 + 2 + 1
    assert costos.costo_por_hora(real, plan, C).tolist() == [0, 6, 2, 1]
    assert costos.costo_medio(real, plan, C) == pytest.approx(9 / 4)


def test_costo_usa_los_supuestos_configurados():
    otro = costos.Costos(faltante=5, sobrante=2)
    assert costos.costo_por_hora([10, 10], [8, 12], otro).tolist() == [10, 4]


def test_costo_rechaza_tamanos_distintos_vacios_y_nan():
    with pytest.raises(ValueError, match="tamaños"):
        costos.costo_por_hora([1, 2], [1], C)
    with pytest.raises(ValueError, match="vacíos"):
        costos.costo_por_hora([1, np.nan], [1, 2], C)
    with pytest.raises(ValueError, match="No hay horas"):
        costos.costo_medio([], [], C)


def test_costos_negativos_o_rango_invalido():
    with pytest.raises(ValueError):
        costos.Costos(faltante=-1, sobrante=1)
    with pytest.raises(ValueError):
        costos.Costos(faltante=3, sobrante=1, factor_minimo=0.5, factor_maximo=0.4)


# -------------------------------------------------------- factor de ajuste

def test_planificar_aplica_el_factor_y_redondea():
    assert costos.planificar([100, 10, 0], 0.15).tolist() == [115, 12, 0]   # 11.5 -> 12
    assert costos.planificar([100], 0.0).tolist() == [100]


def test_factores_cubren_0_a_40_por_ciento():
    f = C.factores()
    assert f[0] == 0.0 and f[-1] == 0.40 and len(f) == 41


def test_mejor_factor_corrige_una_subestimacion_sistematica():
    real = np.full(200, 120.0)
    pred = np.full(200, 100.0)          # el modelo subestima un 20 %
    factor, curva = costos.mejor_factor(real, pred, C)
    assert factor == pytest.approx(0.20)
    assert curva.loc[curva.factor == 0.20, "costo_medio_hora"].item() == 0
    assert curva["costo_medio_hora"].iloc[0] == 60   # sin ajuste: faltan 20 x 3


def test_mejor_factor_es_cero_si_el_modelo_sobreestima():
    factor, _ = costos.mejor_factor(np.full(10, 80.0), np.full(10, 100.0), C)
    assert factor == 0.0


def test_mejor_factor_respeta_el_maximo():
    factor, curva = costos.mejor_factor(np.full(10, 300.0), np.full(10, 100.0), C)
    assert factor == pytest.approx(0.40) and curva.factor.max() == pytest.approx(0.40)


def test_en_empate_elige_el_menor_factor():
    # Con 0 en real y pred, todos los factores dan costo 0.
    factor, _ = costos.mejor_factor(np.zeros(5), np.zeros(5), C)
    assert factor == 0.0


def test_mejor_factor_nunca_empeora_el_costo_sin_ajuste():
    rng = np.random.default_rng(0)
    real = rng.integers(0, 400, 500).astype(float)
    pred = np.rint(real * rng.uniform(0.6, 1.3, 500))
    factor, curva = costos.mejor_factor(real, pred, C)
    assert curva["costo_medio_hora"].min() <= costos.costo_medio(real, pred, C)


# ------------------------------------------------------------- configuración

def test_carga_configs_costos_yaml(entorno):
    c = costos.cargar()
    assert (c.faltante, c.sobrante) == (3, 1)
    assert (c.factor_minimo, c.factor_maximo, c.factor_paso) == (0.0, 0.40, 0.01)


def test_el_archivo_documenta_que_son_supuestos():
    texto = (config.RAIZ / "configs" / "costos.yaml").read_text(encoding="utf-8")
    assert "SUPUESTOS" in texto
    c = yaml.safe_load(texto)
    assert (c["costo_faltante"], c["costo_sobrante"]) == (3, 1)


def test_ruta_costos_se_puede_cambiar_por_entorno(entorno, monkeypatch, tmp_path):
    otro = tmp_path / "otros_costos.yaml"
    otro.write_text("costo_faltante: 10\ncosto_sobrante: 2\n", encoding="utf-8")
    monkeypatch.setenv("BICIS_COSTOS", str(otro))
    assert (costos.cargar().faltante, costos.cargar().sobrante) == (10, 2)
