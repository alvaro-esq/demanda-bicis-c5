"""Evalúa el modelo de un run de MLflow en otro mes y lo registra en ese mismo run.

Carga el modelo registrado (runs:/<run_id>/modelo), lo aplica a las filas de la
fuente del mes pedido (columnas de transformar.py; nunca casual ni registered),
con el mismo posproceso que predecir.py, y lo compara con el mes de validación
del run (recalculado con el mismo modelo).

Registra en el run:
  - métricas <prefijo>_mae, _rmse, _error_medio (predicho - real; > 0 = sobreestima),
    _error_medio_pct, y el MAE y el error medio por hora como series (step = hora);
  - el mes evaluado como dataset con contexto "evaluation";
  - artefactos en evaluacion_<mes>/: comparación por hora (csv y png),
    real vs. predicción del mes y un resumen.json.

Uso:
    python -m bicis.evaluar --run-id <id> --mes 2012-01
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from bicis import config
from bicis.entrenar import git_info, huella, leer_periodo, metricas_error, posprocesar
from bicis.transformar import columnas_salida


def rango_mes(mes: str) -> tuple[str, str]:
    anio, m = (int(x) for x in mes.split("-"))
    return f"{anio:04d}-{m:02d}-01", f"{anio:04d}-{m:02d}-{calendar.monthrange(anio, m)[1]:02d}"


def evaluar_periodo(modelo, desde: str, hasta: str, objetivo: str) -> pd.DataFrame:
    cols_t = columnas_salida()
    df = leer_periodo(desde, hasta, list(dict.fromkeys([*cols_t, objetivo])))
    if df.empty:
        raise ValueError(f"No hay filas en la fuente entre {desde} y {hasta}")
    pred = posprocesar(np.asarray(modelo.predict(df[cols_t]), dtype=float))
    return df.assign(pred=pred, error=pred - df[objetivo])


def resumen(df: pd.DataFrame, objetivo: str) -> dict:
    r = metricas_error(df[objetivo], df["pred"].to_numpy())
    r["cnt_medio"] = float(df[objetivo].mean())
    r["error_medio_pct"] = 100 * r["error_medio"] / r["cnt_medio"]
    r["mae_pct"] = 100 * r["mae"] / r["cnt_medio"]
    r["horas"] = int(len(df))
    r["horas_sobreestimadas_pct"] = float(100 * (df["error"] > 0).mean())
    return r


def por_hora(df: pd.DataFrame, objetivo: str) -> pd.DataFrame:
    return df.groupby("hr").agg(mae=("error", lambda e: e.abs().mean()),
                                error_medio=("error", "mean"),
                                cnt_medio=(objetivo, "mean")).reset_index()


def _graficas(carpeta: Path, comp: pd.DataFrame, ev: pd.DataFrame, et_base: str, et_mes: str,
              objetivo: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = comp["hr"].to_numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ancho = 0.4
    ax1.bar(x - ancho / 2, comp["mae_base"], ancho, label=et_base, color="#1f77b4")
    ax1.bar(x + ancho / 2, comp["mae_mes"], ancho, label=et_mes, color="#ff7f0e")
    ax1.set(title="MAE por hora del día", ylabel="MAE")
    ax1.legend()
    ax2.bar(x - ancho / 2, comp["error_medio_base"], ancho, label=et_base, color="#1f77b4")
    ax2.bar(x + ancho / 2, comp["error_medio_mes"], ancho, label=et_mes, color="#ff7f0e")
    ax2.axhline(0, color="gray", lw=1)
    ax2.set(title="Error medio por hora (> 0: sobreestima, < 0: subestima)",
            xlabel="hora", ylabel="predicho − real", xticks=range(24))
    ax2.legend()
    fig.tight_layout()
    fig.savefig(carpeta / "comparacion_por_hora.png", dpi=120)
    plt.close(fig)

    momento = pd.to_datetime(ev["dteday"]) + pd.to_timedelta(ev["hr"], unit="h")
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.plot(momento, ev[objetivo], lw=0.9, label="real", color="#1f77b4")
    ax.plot(momento, ev["pred"], lw=0.9, label="predicción", color="#ff7f0e", alpha=0.85)
    ax.set(title=f"Real vs. predicción por hora — {et_mes}", ylabel=objetivo)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(carpeta / f"real_vs_prediccion_{et_mes}.png", dpi=120)
    plt.close(fig)


def evaluar(run_id: str, mes: str, registrar: bool = True) -> dict:
    import mlflow

    objetivo = config.params()["modelo"]["objetivo"]
    run = mlflow.get_run(run_id)
    modelo = mlflow.pyfunc.load_model(f"runs:/{run_id}/modelo")

    desde, hasta = rango_mes(mes)
    base_desde, base_hasta = rango_mes(run.data.params["validacion_mes"])
    ev = evaluar_periodo(modelo, desde, hasta, objetivo)
    base = evaluar_periodo(modelo, base_desde, base_hasta, objetivo)

    r_mes, r_base = resumen(ev, objetivo), resumen(base, objetivo)
    comp = por_hora(base, objetivo).merge(por_hora(ev, objetivo), on="hr", suffixes=("_base", "_mes"))
    sentido = "sobreestima" if r_mes["error_medio"] > 0 else "subestima"
    resultado = {
        "run_id": run_id, "mes": mes, "comparado_con": run.data.params["validacion_mes"],
        "mes_evaluado": r_mes, "validacion": r_base, "sentido": sentido,
        "por_hora": comp.round(3).to_dict(orient="records"),
    }
    if not registrar:
        return resultado

    pref = "eval_" + mes.replace("-", "_")
    git = git_info()
    fuente = str(config.ruta("fuente_db").resolve())
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"{pref}_{k}": v for k, v in r_mes.items()})
        for fila in comp.itertuples():
            mlflow.log_metric(f"{pref}_mae_hora", fila.mae_mes, step=int(fila.hr))
            mlflow.log_metric(f"{pref}_error_medio_hora", fila.error_medio_mes, step=int(fila.hr))
            mlflow.log_metric("val_mae_hora", fila.mae_base, step=int(fila.hr))
            mlflow.log_metric("val_error_medio_hora", fila.error_medio_base, step=int(fila.hr))
        mlflow.set_tags({
            f"{pref}_sentido": sentido,
            f"{pref}_git_commit": git["commit"],
            f"{pref}_huella_datos": huella(ev.drop(columns=["pred", "error"])),
        })
        ds = mlflow.data.from_pandas(ev.drop(columns=["pred", "error"]), source=fuente,
                                     targets=objetivo, name=f"evaluacion_{desde}_{hasta}")
        mlflow.log_input(ds, context="evaluation")
        with tempfile.TemporaryDirectory() as tmp:
            carpeta = Path(tmp)
            comp.round(3).to_csv(carpeta / "comparacion_por_hora.csv", index=False)
            (carpeta / "resumen.json").write_text(
                json.dumps({k: v for k, v in resultado.items() if k != "por_hora"},
                           indent=2, ensure_ascii=False), encoding="utf-8")
            _graficas(carpeta, comp, ev, run.data.params["validacion_mes"], mes, objetivo)
            mlflow.log_artifacts(str(carpeta), artifact_path=f"evaluacion_{mes}")
    return resultado


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evalúa el modelo de un run en otro mes.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mes", required=True, help="AAAA-MM")
    parser.add_argument("--sin-registrar", action="store_true")
    args = parser.parse_args(argv)
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        parser.error("Define MLFLOW_TRACKING_URI (o usa make evaluar)")

    r = evaluar(args.run_id, args.mes, registrar=not args.sin_registrar)
    for nombre, x in ((r["comparado_con"], r["validacion"]), (r["mes"], r["mes_evaluado"])):
        print(f"{nombre}: MAE {x['mae']:.2f} ({x['mae_pct']:.0f}% del cnt medio {x['cnt_medio']:.1f}) · "
              f"RMSE {x['rmse']:.2f} · error medio {x['error_medio']:+.2f} ({x['error_medio_pct']:+.0f}%) · "
              f"{x['horas']} horas")
    print(f"En {r['mes']} el modelo {r['sentido']}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
