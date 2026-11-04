"""Compara escalones de complejidad y elige el ganador en validación.

Escalones (params.yaml -> escalones.orden):
  1. perfil_historico        promedio de cnt por hora y tipo de día
  2. ridge                   hr y mnth como categorías
  3. random_forest           el actual (se puede reutilizar un run existente)
  4. hist_gradient_boosting  objetivo en log; búsqueda de learning_rate en runs
                             anidados bajo un run padre

Cada escalón es un run con su propio factor de ajuste. Gana el de menor costo
medio por hora en validación (ya ajustado con su factor) y, en empate, el de
menor MAE de validación. Nunca se usa 2012: la función falla si la ventana de
entrenamiento o de validación toca 2012 o después.

Ninguno de estos entrenamientos cambia models/modelo.joblib ni metrics.json.

Uso:
    python -m bicis.escalones [--reusar-run-rf <run_id>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bicis import config
from bicis.entrenar import entrenar, huella, leer_periodo
from bicis.transformar import columnas_salida

LIMITE = "2012-01-01"  # nada de 2012 en adelante participa de la elección


def verificar_sin_2012(p: dict) -> None:
    m = p["modelo"]
    for nombre in ("entrenamiento", "validacion"):
        if m[nombre]["hasta"] >= LIMITE:
            raise ValueError(f"La ventana de {nombre} ({m[nombre]}) usa datos de 2012: "
                             "no se puede usar para elegir el modelo")


def clave_eleccion(r: dict) -> tuple[float, float]:
    """Menor costo medio por hora (con su factor); en empate, menor MAE."""
    return (round(r["costo"], 6), round(r["mae"], 6))


def _resultado(escalon: int, tipo: str, run_id: str, metricas: dict) -> dict:
    return {
        "escalon": escalon, "tipo": tipo, "run_id": run_id,
        "costo": metricas["val_costo_medio_hora_ajustado"],
        "costo_sin_ajuste": metricas["val_costo_medio_hora"],
        "factor": metricas["factor_ajuste"],
        "mae": metricas["val_mae"], "rmse": metricas["val_rmse"],
        "error_medio": metricas["val_error_medio"],
        "latencia_p95_ms_dia": metricas["latencia_p95_ms_dia"],
        "modelo_tamano_mb": metricas["modelo_tamano_mb"],
    }


def _metricas_run(run_id: str) -> dict:
    import mlflow
    return mlflow.get_run(run_id).data.metrics


def _validar_run_reutilizado(run_id: str, p: dict) -> None:
    """El run reutilizado debe tener la misma ventana, costos y datos que los demás."""
    import mlflow
    from bicis import costos as mod_costos

    run = mlflow.get_run(run_id)
    pr, tags, m = run.data.params, run.data.tags, p["modelo"]
    cst = mod_costos.cargar()
    esperado = {
        "modelo_tipo": "random_forest",
        "entrenamiento_inicio": m["entrenamiento"]["desde"],
        "entrenamiento_fin": m["entrenamiento"]["hasta"],
        "validacion_mes": m["validacion"]["desde"][:7],
        "costo_faltante": str(cst.faltante), "costo_sobrante": str(cst.sobrante),
    }
    distintos = {k: (pr.get(k), v) for k, v in esperado.items() if pr.get(k) != v}
    cols = list(dict.fromkeys([*columnas_salida(), m["objetivo"]]))
    huella_actual = huella(leer_periodo(m["entrenamiento"]["desde"], m["entrenamiento"]["hasta"], cols))
    if tags.get("huella_datos_entrenamiento") != huella_actual:
        distintos["huella_datos_entrenamiento"] = (tags.get("huella_datos_entrenamiento"), huella_actual)
    if tags.get("features") != ",".join(config.features()):
        distintos["features"] = (tags.get("features"), ",".join(config.features()))
    if distintos:
        raise ValueError(f"El run {run_id} no es comparable (registrado, actual): {distintos}")


def ejecutar(reusar_run_rf: str | None = None) -> dict:
    import mlflow

    p = config.params()
    verificar_sin_2012(p)
    mlflow.set_experiment(p["mlflow"]["experimento"])
    grupo = "escalones-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    resultados = []

    for escalon, tipo in enumerate(p["escalones"]["orden"], start=1):
        tags = {"grupo": grupo, "escalon": str(escalon)}
        busqueda = p["escalones"].get("busqueda", {}).get(tipo)

        if tipo == "random_forest" and reusar_run_rf:
            _validar_run_reutilizado(reusar_run_rf, p)
            mlflow.MlflowClient().set_tag(reusar_run_rf, "grupo", grupo)
            mlflow.MlflowClient().set_tag(reusar_run_rf, "escalon", str(escalon))
            resultados.append(_resultado(escalon, tipo, reusar_run_rf, _metricas_run(reusar_run_rf)))
            continue

        if not busqueda:
            m = entrenar(tipo, guardar=False, etiquetas=tags, nombre_run=f"{escalon}-{tipo}")
            resultados.append(_resultado(escalon, tipo, m["mlflow_run_id"], _metricas_run(m["mlflow_run_id"])))
            continue

        # Búsqueda: un run padre y un run anidado por combinación.
        (param, valores), = busqueda.items()
        with mlflow.start_run(run_name=f"{escalon}-{tipo}-busqueda-{param}") as padre:
            mlflow.set_tags({**tags, "busqueda": param})
            mlflow.log_param(f"busqueda_{param}", ",".join(map(str, valores)))
            hijos = []
            for v in valores:
                m = entrenar(tipo, guardar=False, hiperparametros={param: v}, anidado=True,
                             etiquetas={**tags, "busqueda": param}, nombre_run=f"{tipo}-{param}={v}")
                hijos.append({param: v, **_resultado(escalon, tipo, m["mlflow_run_id"],
                                                     _metricas_run(m["mlflow_run_id"]))})
            mejor = min(hijos, key=clave_eleccion)
            mlflow.log_param(f"mejor_{param}", mejor[param])
            mlflow.set_tag("mejor_run_id", mejor["run_id"])
            mlflow.log_metrics({
                "val_costo_medio_hora_ajustado": mejor["costo"], "factor_ajuste": mejor["factor"],
                "val_mae": mejor["mae"], "val_costo_medio_hora": mejor["costo_sin_ajuste"],
            })
            with tempfile.TemporaryDirectory() as tmp:
                ruta = Path(tmp) / "busqueda.csv"
                pd.DataFrame(hijos).to_csv(ruta, index=False)
                mlflow.log_artifact(str(ruta))
            mlflow.MlflowClient().set_tag(mejor["run_id"], "mejor_de_busqueda", "true")
        resultados.append({**mejor, "run_padre": padre.info.run_id, "busqueda": hijos})

    ganador = min(resultados, key=clave_eleccion)
    cliente = mlflow.MlflowClient()
    for r in resultados:
        cliente.set_tag(r["run_id"], "ganador_escalones", str(r is ganador).lower())

    tabla = pd.DataFrame([{k: v for k, v in r.items() if k != "busqueda"} for r in resultados])
    with mlflow.start_run(run_name=f"comparacion-{grupo}"):
        mlflow.set_tags({"grupo": grupo, "tipo_run": "comparacion",
                         "ganador_run_id": ganador["run_id"], "ganador_tipo": ganador["tipo"],
                         "regla": "min costo medio por hora (con factor) en validación; empate: min MAE",
                         "datos_2012": "no usados"})
        with tempfile.TemporaryDirectory() as tmp:
            tabla.to_csv(Path(tmp) / "comparacion.csv", index=False)
            mlflow.log_artifact(str(Path(tmp) / "comparacion.csv"))
    return {"grupo": grupo, "ganador": ganador, "resultados": resultados, "tabla": tabla}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entrena y compara los escalones de complejidad.")
    parser.add_argument("--reusar-run-rf", help="run existente del Random Forest a reutilizar")
    args = parser.parse_args(argv)
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        parser.error("Define MLFLOW_TRACKING_URI (o usa make escalones)")
    r = ejecutar(args.reusar_run_rf)
    cols = ["escalon", "tipo", "costo", "factor", "costo_sin_ajuste", "mae", "error_medio", "run_id"]
    print(r["tabla"][cols].round(3).to_string(index=False))
    for b in r["resultados"]:
        if "busqueda" in b:
            print(pd.DataFrame(b["busqueda"])[["learning_rate", "costo", "factor", "mae", "run_id"]]
                  .round(3).to_string(index=False))
    g = r["ganador"]
    print(f"Ganador: {g['tipo']} (run {g['run_id']}) · costo {g['costo']:.2f}/hora con factor "
          f"{100 * g['factor']:.0f} % · MAE {g['mae']:.2f}")
    print(json.dumps({"grupo": r["grupo"], "ganador_run_id": g["run_id"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
