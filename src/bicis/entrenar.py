"""Entrena el modelo de cnt, mide el error en validación y registra el run en MLflow.

Entrena con modelo.entrenamiento (enero-noviembre de 2011) y valida con
modelo.validacion (diciembre de 2011), leyendo la fuente en solo lectura.
Nunca lee casual ni registered.

Guarda el modelo en rutas.modelo (el que usa predecir.py) y las métricas en
rutas.metricas. Además, cada ejecución es un run del experimento
mlflow.experimento con:
  - parámetros: tipo de modelo, hiperparámetros y ventana de datos;
  - métricas de validación: MAE, RMSE, error medio (positivo = sobreestima),
    latencia p95 de predecir un día (24 horas) y tamaño del modelo;
  - artefactos: real vs. predicción en validación, MAE por hora e importancia
    de las features;
  - linaje: commit de Git, features, huella de los datos y los datasets de
    entrenamiento y validación;
  - el modelo (skops), con firma: un DataFrame con las columnas de transformar.py.

Uso:
    python -m bicis.entrenar [--modelo {random_forest,extra_trees,gradient_boosting}] [--sin-mlflow]

El servidor de MLflow sale de MLFLOW_TRACKING_URI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from bicis import config
from bicis.transformar import columnas_salida

MODELOS = {
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}

# Tipos que skops no carga por defecto y que confiamos porque el archivo lo
# generamos nosotros. Es el único que necesitan los tres modelos; no se agrega
# nada más de forma automática.
TIPOS_CONFIABLES_SKOPS = ["sklearn.tree._tree.Tree"]


# --------------------------------------------------------------------- datos

def leer_periodo(desde: str, hasta: str, columnas: list[str]) -> pd.DataFrame:
    if (config.prohibidas() - {"dteday"}).intersection(columnas):
        raise ValueError(f"entrenar no debe leer columnas prohibidas: {columnas}")
    consulta = (
        f"SELECT {', '.join(columnas)} FROM {config.tabla_fuente()} "
        "WHERE dteday BETWEEN ? AND ? ORDER BY dteday, hr"
    )
    con = config.conectar_fuente()
    try:
        return pd.read_sql_query(consulta, con, params=(desde, hasta))
    finally:
        con.close()


def huella(df: pd.DataFrame) -> str:
    """SHA-256 del contenido (columnas, tipos y valores) de un DataFrame."""
    h = hashlib.sha256()
    h.update(json.dumps([(c, str(t)) for c, t in df.dtypes.items()]).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


# -------------------------------------------------------------------- modelo

def construir(tipo: str, hiperparametros: dict):
    if tipo not in MODELOS:
        raise ValueError(f"Modelo desconocido {tipo!r}. Opciones: {sorted(MODELOS)}")
    return MODELOS[tipo](**hiperparametros)


def posprocesar(pred: np.ndarray) -> np.ndarray:
    """Igual que predecir.py: recorte a >= 0 y redondeo a entero."""
    return np.rint(np.clip(pred, 0, None))


def pipeline_registrable(modelo, X_transformar: pd.DataFrame, feats: list[str]) -> Pipeline:
    """Envuelve el modelo ya entrenado para que acepte la salida de transformar.py.

    El selector deja solo las features (descarta dteday) y no reentrena el modelo.
    """
    selector = ColumnTransformer(
        [("features", "passthrough", feats)], remainder="drop", verbose_feature_names_out=False
    ).set_output(transform="pandas")
    selector.fit(X_transformar)
    return Pipeline([("seleccion", selector), ("modelo", modelo)])


def metricas_error(real: pd.Series, pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(real, pred)),
        "rmse": float(np.sqrt(mean_squared_error(real, pred))),
        "error_medio": float(np.mean(pred - real.to_numpy())),  # > 0: sobreestima
    }


def latencia_p95_ms(predictor, X_dia: pd.DataFrame, repeticiones: int) -> float:
    for _ in range(3):  # calentamiento
        predictor.predict(X_dia)
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        predictor.predict(X_dia)
        tiempos.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(tiempos, 95))


def importancias(modelo, feats: list[str]) -> pd.DataFrame:
    return (pd.DataFrame({"feature": feats, "importancia": modelo.feature_importances_})
            .sort_values("importancia", ascending=False, ignore_index=True))


# ---------------------------------------------------------------- artefactos

def graficas(carpeta: Path, val: pd.DataFrame, pred: np.ndarray, mae_hora: pd.DataFrame,
             imp: pd.DataFrame, objetivo: str, etiqueta_val: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    momento = pd.to_datetime(val["dteday"]) + pd.to_timedelta(val["hr"], unit="h")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [3, 1]})
    ax1.plot(momento, val[objetivo], lw=0.9, label="real", color="#1f77b4")
    ax1.plot(momento, pred, lw=0.9, label="predicción", color="#ff7f0e", alpha=0.85)
    ax1.set(title=f"Real vs. predicción por hora — {etiqueta_val}", ylabel=objetivo)
    ax1.legend(loc="upper right")
    tope = max(val[objetivo].max(), pred.max()) * 1.05
    ax2.scatter(val[objetivo], pred, s=6, alpha=0.4)
    ax2.plot([0, tope], [0, tope], color="gray", lw=1, ls="--")
    ax2.set(xlim=(0, tope), ylim=(0, tope), xlabel="real", ylabel="predicción", title="Dispersión")
    fig.tight_layout()
    fig.savefig(carpeta / "real_vs_prediccion.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(mae_hora["hr"], mae_hora["mae"], color="#1f77b4")
    ax.set(title=f"MAE por hora del día — {etiqueta_val}", xlabel="hora", ylabel="MAE",
           xticks=range(24))
    fig.tight_layout()
    fig.savefig(carpeta / "mae_por_hora.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    orden = imp.iloc[::-1]
    ax.barh(orden["feature"], orden["importancia"], color="#1f77b4")
    ax.set(title="Importancia de las features", xlabel="importancia")
    fig.tight_layout()
    fig.savefig(carpeta / "importancia_features.png", dpi=120)
    plt.close(fig)


def git_info() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=config.RAIZ, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "rama": git("rev-parse", "--abbrev-ref", "HEAD"),
                "sucio": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "desconocido", "rama": "desconocida", "sucio": None}


# ------------------------------------------------------------- entrenamiento

def entrenar(tipo: str | None = None, registrar: bool = True) -> dict:
    p = config.params()
    m = p["modelo"]
    tipo = tipo or m["tipo"]
    hp = m["hiperparametros"][tipo]
    feats = config.features()
    objetivo = m["objetivo"]
    cols_t = columnas_salida()  # columnas que produce transformar.py
    cols = list(dict.fromkeys([*cols_t, objetivo]))
    ent_v, val_v = m["entrenamiento"], m["validacion"]

    ent = leer_periodo(ent_v["desde"], ent_v["hasta"], cols)
    val = leer_periodo(val_v["desde"], val_v["hasta"], cols)
    if ent.empty or val.empty:
        raise ValueError("No hay filas en el periodo de entrenamiento o de validación")

    modelo = construir(tipo, hp).fit(ent[feats], ent[objetivo])
    pred = posprocesar(modelo.predict(val[feats]))
    error = metricas_error(val[objetivo], pred)

    # Referencia ingenua: promedio de entrenamiento por (workingday, hr).
    base = ent.groupby(["workingday", "hr"])[objetivo].mean()
    mae_base = mean_absolute_error(val[objetivo], val.set_index(["workingday", "hr"]).index.map(base))

    pipe = pipeline_registrable(modelo, ent[cols_t], feats)
    dias_completos = val.groupby("dteday").size().loc[lambda s: s == 24].index
    X_dia = val.loc[val["dteday"] == dias_completos[0], cols_t].reset_index(drop=True)
    latencia = latencia_p95_ms(pipe, X_dia, p.get("mlflow", {}).get("repeticiones_latencia", 50))

    ruta_modelo = config.ruta("modelo")
    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta_modelo, compress=3)
    tamano_mb = ruta_modelo.stat().st_size / 2**20

    errores = val.assign(pred=pred, abs_err=np.abs(pred - val[objetivo]))
    mae_hora = errores.groupby("hr", as_index=False)["abs_err"].mean().rename(columns={"abs_err": "mae"})
    imp = importancias(modelo, feats)
    huellas = {"entrenamiento": huella(ent), "validacion": huella(val)}
    git = git_info()

    metricas = {
        "modelo_tipo": tipo,
        "mae_validacion": round(error["mae"], 3),
        "rmse_validacion": round(error["rmse"], 3),
        "error_medio_validacion": round(error["error_medio"], 3),
        "mae_referencia_promedio_hora": round(float(mae_base), 3),
        "cnt_medio_validacion": round(float(val[objetivo].mean()), 3),
        "latencia_p95_ms_dia": round(latencia, 2),
        "modelo_tamano_mb": round(tamano_mb, 2),
        "filas_entrenamiento": int(len(ent)),
        "filas_validacion": int(len(val)),
        "entrenamiento": ent_v,
        "validacion": val_v,
        "features": feats,
        "hiperparametros": hp,
        "huella_datos_entrenamiento": huellas["entrenamiento"],
    }
    config.ruta("metricas").write_text(json.dumps(metricas, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")

    if registrar:
        metricas["mlflow_run_id"] = _registrar(
            p, tipo, modelo, pipe, ent, val, pred, error, mae_base, latencia, tamano_mb,
            mae_hora, imp, huellas, git, feats, cols_t, objetivo,
        )
    return metricas


def _registrar(p, tipo, modelo, pipe, ent, val, pred, error, mae_base, latencia, tamano_mb,
               mae_hora, imp, huellas, git, feats, cols_t, objetivo) -> str:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    m = p["modelo"]
    ent_v, val_v = m["entrenamiento"], m["validacion"]
    mes_val = val_v["desde"][:7]
    fuente = str(config.ruta("fuente_db").resolve())

    mlflow.set_experiment(p["mlflow"]["experimento"])
    with mlflow.start_run(run_name=f"{tipo}-{mes_val}") as run:
        # 1. Parámetros
        mlflow.log_params({
            "modelo_tipo": tipo,
            "modelo_clase": type(modelo).__name__,
            **{f"hp_{k}": v for k, v in modelo.get_params().items()},
            "entrenamiento_inicio": ent_v["desde"],
            "entrenamiento_fin": ent_v["hasta"],
            "validacion_mes": mes_val,
            "objetivo": objetivo,
            "n_features": len(feats),
        })

        # 2. Métricas de validación
        mlflow.log_metrics({
            "val_mae": error["mae"],
            "val_rmse": error["rmse"],
            "val_error_medio": error["error_medio"],
            "val_mae_referencia": float(mae_base),
            "latencia_p95_ms_dia": latencia,
            "modelo_tamano_mb": tamano_mb,
            "filas_entrenamiento": len(ent),
            "filas_validacion": len(val),
        })

        # 4. Linaje
        mlflow.set_tags({
            "mlflow.source.git.commit": git["commit"],
            "git_commit": git["commit"],
            "git_rama": git["rama"],
            "git_sucio": str(git["sucio"]),
            "features": ",".join(feats),
            "huella_datos_entrenamiento": huellas["entrenamiento"],
            "huella_datos_validacion": huellas["validacion"],
            "fuente": fuente,
        })
        for nombre, df, contexto, ventana in (("entrenamiento", ent, "training", ent_v),
                                              ("validacion", val, "validation", val_v)):
            ds = mlflow.data.from_pandas(
                df, source=fuente, targets=objetivo,
                name=f"{nombre}_{ventana['desde']}_{ventana['hasta']}",
            )
            mlflow.log_input(ds, context=contexto)

        # 3. Artefactos
        with tempfile.TemporaryDirectory() as tmp:
            carpeta = Path(tmp)
            graficas(carpeta, val, pred, mae_hora, imp, objetivo, f"validación {mes_val}")
            mae_hora.round(3).to_csv(carpeta / "mae_por_hora.csv", index=False)
            imp.round(5).to_csv(carpeta / "importancia_features.csv", index=False)
            (carpeta / "linaje.json").write_text(json.dumps({
                "git": git, "features": feats, "columnas_entrada": cols_t,
                "huellas_sha256": huellas, "fuente": fuente,
                "ventana": {"entrenamiento": ent_v, "validacion": val_v},
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            mlflow.log_artifacts(str(carpeta), artifact_path="evaluacion")

        # 5. Modelo con firma: entra un DataFrame con las columnas de transformar.py
        ejemplo = val[cols_t].head(24).reset_index(drop=True)
        firma = infer_signature(ejemplo, pipe.predict(ejemplo))
        mlflow.sklearn.log_model(
            pipe,
            name="modelo",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
            skops_trusted_types=TIPOS_CONFIABLES_SKOPS,
            signature=firma,
            input_example=ejemplo,
        )
        return run.info.run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entrena el modelo de cnt y lo registra en MLflow.")
    parser.add_argument("--modelo", choices=sorted(MODELOS), default=None,
                        help="tipo de modelo (por defecto, modelo.tipo de params.yaml)")
    parser.add_argument("--sin-mlflow", action="store_true", help="no registra el run en MLflow")
    args = parser.parse_args(argv)

    if not args.sin_mlflow and not os.environ.get("MLFLOW_TRACKING_URI"):
        parser.error("Define MLFLOW_TRACKING_URI (o usa make entrenar) o pasa --sin-mlflow")

    m = entrenar(args.modelo, registrar=not args.sin_mlflow)
    print(f"{m['modelo_tipo']}: MAE {m['mae_validacion']} · RMSE {m['rmse_validacion']} · "
          f"error medio {m['error_medio_validacion']:+} · p95 {m['latencia_p95_ms_dia']} ms/día · "
          f"{m['modelo_tamano_mb']} MB")
    print(f"Modelo   -> {config.ruta('modelo')}")
    print(f"Métricas -> {config.ruta('metricas')}")
    if "mlflow_run_id" in m:
        print(f"MLflow   -> run {m['mlflow_run_id']} en {os.environ['MLFLOW_TRACKING_URI']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
