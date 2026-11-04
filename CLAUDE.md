# demanda-bicis-c5 — contrato del proyecto

Cada mañana el pipeline toma una fecha `F` (AAAA-MM-DD) y predice cuántas
bicicletas se alquilarán (`cnt`) en cada una de las 24 horas de `F`.

Este proyecto es independiente de `demanda-bicis`: no comparte código, repositorio
de GitHub ni remoto DVC con él.

## Estructura

```
src/bicis/      paquete del pipeline (los cinco pasos, config.py y entrenar.py)
scripts/        utilidades fuera del pipeline (crear_fuente.py)
tests/          pruebas con pytest
dags/           DAGs de Airflow
models/         modelo entrenado (versionado con DVC, no con git)
airflow/        docker-compose.yaml, Dockerfile y .env.example de Airflow
fuente/         bicis.db: sistema de origen simulado (no se versiona)
data/           staging/ y salidas/ (no se versiona)
```

## Fuente de datos (solo lectura)

- `fuente/bicis.db`, tabla `registros_horarios`: las columnas de `hour.csv` del
  dataset Bike Sharing de UCI, con clave `(dteday, hr)` y `dteday` en texto ISO.
- La crea **únicamente** `scripts/crear_fuente.py` (`make fuente`). Simula el
  sistema de origen: el pipeline **nunca** escribe en ella. Hay que abrirla en solo
  lectura (`sqlite3.connect("file:...?mode=ro", uri=True)`). En Airflow además está
  montada como `:ro`.
- La fuente tiene horas faltantes en algunos días: no todos los días traen 24 filas.

## Configuración: rutas

Las rutas salen de variables de entorno. Si una variable no está definida, se usa el
valor por defecto de `params.yaml`. Las rutas relativas se resuelven desde la
carpeta que contiene `params.yaml`, no desde el directorio de trabajo.

| Variable            | Clave en params.yaml | Por defecto                    |
|---------------------|----------------------|--------------------------------|
| `BICIS_PARAMS`      | —                    | `params.yaml` de la raíz       |
| `BICIS_ZIP_ORIGEN`  | `rutas.zip_origen`   | `bike_sharing_dataset.zip`     |
| `BICIS_FUENTE_DB`   | `rutas.fuente_db`    | `fuente/bicis.db`              |
| `BICIS_STAGING_DIR` | `rutas.staging_dir`  | `data/staging`                 |
| `BICIS_SALIDAS_DB`  | `rutas.salidas_db`   | `data/salidas/predicciones.db` |
| `BICIS_MODELO`      | `rutas.modelo`       | `models/modelo.joblib`         |
| `BICIS_METRICAS`    | `rutas.metricas`     | `metrics.json`                 |

- La resolución de rutas vive en `src/bicis/config.py`, un módulo auxiliar que no es
  un paso del pipeline. Ningún módulo escribe rutas a mano ni lee `os.environ`
  directamente: todos usan `config`.
- Los hiperparámetros, la lista de features y el año de entrenamiento salen de
  `params.yaml`; no se repiten en el código.

## Pipeline: cinco módulos en `src/bicis/`

`extraer` → `limpiar` → `transformar` → `predecir` → `almacenar`

Cada módulo expone una sola función pública:

```python
def ejecutar(fecha: str) -> pathlib.Path: ...
```

- `fecha` es `"AAAA-MM-DD"`. Un formato inválido es un error.
- Lee **solo** el archivo del paso anterior en `data/staging/<fecha>/` (`extraer` lee
  la fuente y `predecir` lee además el modelo). Escribe su propio archivo y devuelve
  su ruta.
- Si falta el archivo del paso anterior, lanza `FileNotFoundError`. No lo regenera.
- Es idempotente: si se vuelve a ejecutar con la misma fecha, sobrescribe su salida.
- No hay estado compartido entre pasos fuera de esos archivos. Así Airflow puede
  correr cada paso como una tarea separada y reintentarla sola.

| Paso          | Lee                                   | Escribe                                  |
|---------------|---------------------------------------|------------------------------------------|
| `extraer`     | `fuente/bicis.db` (filas de `fecha`)  | `data/staging/<fecha>/extraido.csv`      |
| `limpiar`     | `extraido.csv`                        | `data/staging/<fecha>/limpio.csv`        |
| `transformar` | `limpio.csv`                          | `data/staging/<fecha>/features.csv`      |
| `predecir`    | `features.csv` + `models:/demanda-bicis-c5@champion` | `data/staging/<fecha>/predicciones.csv` |
| `almacenar`   | `predicciones.csv`                    | `data/salidas/predicciones.db`           |

### Responsabilidades

- **extraer**: selecciona las filas con `dteday = fecha`, pero solo las columnas que
  se conocen por la mañana: `dteday`, `hr` y las features de `params.yaml`. **No
  extrae `cnt`, `casual` ni `registered`.** Si la fecha no tiene filas, escribe el
  archivo solo con el encabezado: quien falla es `limpiar`.
- **limpiar**: falla (`ValueError`) si el día no tiene filas. Revisa columnas (las que
  faltan y las prohibidas), tipos, rangos válidos (`hr` 0–23, `weathersit` 1–4, `temp`,
  `atemp`, `hum` y `windspeed` en [0, 1]) y exactamente 24 filas, `hr` 0–23. Una hora
  que falta se agrega con las columnas de calendario del día y el clima de la hora
  vecina más cercana, y se marca con `hora_imputada = 1`.
- **transformar**: deja `dteday`, `hr` y las columnas de `modelo.features`, en ese
  orden. Lanza un error si aparece alguna columna de `modelo.prohibidas`.
- **predecir**: carga `models:/demanda-bicis-c5@champion` desde MLflow
  (`MLFLOW_TRACKING_URI`), resolviendo el alias a una versión exacta. Predice `cnt`, lo
  recorta a ≥ 0, lo redondea y aplica el factor de ajuste del tag `factor_ajuste` de
  esa versión: `cnt_predicho = round(pred × (1 + factor))`. Escribe `fecha, hr,
  cnt_predicho, version_modelo` (por ejemplo, `demanda-bicis-c5/1`). **Nunca entrena.**
- **almacenar**: guarda en `data/salidas/predicciones.db`, tabla `predicciones`
  (`fecha TEXT, hr INTEGER, cnt_predicho INTEGER, creado_en TEXT, version_modelo TEXT`,
  PK `(fecha, hr)`). Si la base es anterior a `version_modelo`, se migra con `ALTER TABLE`
  (las filas viejas quedan en NULL). **Reemplaza el día completo**: en una sola transacción borra todas
  las filas de `fecha` e inserta las 24 nuevas. Devuelve la ruta de la base.

## Modelo

- `RandomForestRegressor` de scikit-learn para predecir `cnt` (`modelo.tipo` en
  `params.yaml`). Los hiperparámetros de cada tipo están en `modelo.hiperparametros`.
  `--modelo extra_trees|gradient_boosting` permite comparar otros tipos, pero el modelo
  de producción es el Random Forest.
- Se entrena con **enero a noviembre de 2011** y el MAE se mide en **diciembre de
  2011** (`modelo.entrenamiento` y `modelo.validacion` en `params.yaml`), leyendo de la
  fuente en solo lectura. El modelo que se guarda es el entrenado con enero–noviembre.
- `src/bicis/entrenar.py` (`make entrenar [MODELO=...]`) produce `models/modelo.joblib`
  (el modelo pelado que usa `predecir.py`, versionado con DVC) y `metrics.json` (en git).
- Cada ejecución es un run del experimento `demanda-bicis-c5` en MLflow
  (`MLFLOW_TRACKING_URI`, `http://localhost:5000` en el Makefile). El run registra:
  - **Parámetros**: tipo de modelo, todos los hiperparámetros (`hp_*`) y la ventana
    (`entrenamiento_inicio`, `entrenamiento_fin`, `validacion_mes`).
  - **Métricas**: `val_mae`, `val_rmse` y `val_error_medio` (predicho − real; > 0 significa
    que sobreestima), `val_mae_referencia`, `latencia_p95_ms_dia` (predecir 24 horas) y
    `modelo_tamano_mb`. Se calculan con el mismo posproceso que `predecir.py`.
  - **Artefactos** en `evaluacion/`: real vs. predicción en diciembre, MAE por hora (png y
    csv), importancia de las features (png y csv) y `linaje.json`.
  - **Linaje**: tags con el commit de Git (y si el árbol estaba sucio), las features y el
    SHA-256 de los datos. Entrenamiento y validación se registran como datasets
    (`training` y `validation`).
  - **Modelo** (`modelo/`): un Pipeline que primero se queda con las features y luego
    aplica el regresor ya entrenado. Su firma es un DataFrame con las columnas de
    `transformar.py` (incluida `dteday`). Se serializa con skops, confiando solo en
    `sklearn.tree._tree.Tree`.
- `--sin-mlflow` entrena sin registrar el run.
- `src/bicis/evaluar.py` (`make evaluar RUN_ID=... MES=AAAA-MM`) carga el modelo de un run,
  lo evalúa en otro mes de la fuente y lo compara con el mes de validación. Registra todo
  en el **mismo** run: métricas `eval_<mes>_*`, MAE y error medio por hora como series
  (step = hora), el mes como dataset `evaluation` y artefactos en `evaluacion_<mes>/`.

## Escalones y modelo champion

- `src/bicis/escalones.py` (`make escalones [RUN_RF=<id>]`) entrena un run por escalón,
  con su propio factor: `perfil_historico` (promedio por hora y tipo de día), `ridge`
  (`hr` y `mnth` one-hot), `random_forest` y `hist_gradient_boosting` (objetivo en
  `log1p`). La búsqueda de `learning_rate` ∈ [0.05, 0.1, 0.2] va en runs anidados bajo
  un run padre. Gana el menor `val_costo_medio_hora_ajustado` y, en empate, el menor
  `val_mae`. Falla si alguna ventana toca 2012. No modifica `models/modelo.joblib` ni
  `metrics.json`.
- `src/bicis/registro.py` registra un run como una versión de `demanda-bicis-c5`, le pone
  el alias `champion` y guarda como tags de la versión `factor_ajuste`, el costo (con y sin
  ajuste), `val_mae`, la ventana, las features, el run y el commit. La primera
  promoción (versión 1, HGB con lr 0.1) se hizo a mano (`promocion=manual`); las
  siguientes no deberían.
- `models/modelo.joblib` sigue saliendo de `make entrenar` y está versionado con DVC,
  pero **predecir ya no lo usa**: el modelo de producción es el champion del registro.

## Costo del operador

- `configs/costos.yaml` (`rutas.costos`, `BICIS_COSTOS`) tiene **supuestos del caso**, no
  costos medidos: cada bicicleta que faltó cuesta 3 y cada una que sobró cuesta 1.
- La lógica está en `src/bicis/costos.py`. Por hora, `costo = 3 × max(real − plan, 0) +
  1 × max(plan − real, 0)`, y la métrica es el costo medio por hora.
- El operador planifica con `round(predicción × (1 + factor))`. El factor se elige en
  validación entre 0 % y 40 % (paso 1 %), como el que minimiza el costo medio por hora;
  en empate, se elige el menor.
- Cada run registra `val_costo_medio_hora` (sin ajuste), `factor_ajuste`,
  `val_costo_medio_hora_ajustado`, `val_ahorro_costo_pct`, la curva
  (`val_costo_medio_hora_por_factor`, step = factor en %), los supuestos como parámetros
  y los artefactos `evaluacion/costo_vs_factor.{png,csv}` y `evaluacion/costos.yaml`.
- `predecir.py` no aplica el factor: la predicción sigue siendo la del modelo.
- **Nunca usar `casual` ni `registered`**: ni como feature, ni para derivar features,
  ni para calcular el objetivo. Son los componentes de `cnt` (`cnt = casual +
  registered`), así que usarlos filtraría el objetivo. Las pruebas deben verificarlo.
- Tampoco son features `instant` ni `dteday` (identificadores), ni `yr` (es constante
  en el año de entrenamiento, así que el modelo no puede aprender nada de ella).
- El clima del día se usa como si fuera el pronóstico disponible por la mañana.

## Versionado

- git: código, `params.yaml`, `metrics.json` y los archivos `.dvc`.
- DVC: `models/modelo.joblib` (`dvc add` + `dvc push` después de reentrenar). Remoto por defecto `local` →
  `/home/andres/dvc-storage/demanda-bicis-c5`.
- No se versionan: el `.zip`, `fuente/`, `data/` ni `.venv/` (todos se pueden
  regenerar).

## Airflow

- `airflow/docker-compose.yaml` (Airflow 3.3.2, LocalExecutor, Postgres) usa una
  imagen propia que instala `requirements.txt`.
- El repositorio se monta en `/opt/proyecto`, fuera de `/opt/airflow`, y encima
  `fuente/` se monta en `/opt/proyecto/fuente` en **solo lectura**. Los DAGs se leen de
  `/opt/proyecto/dags` y `PYTHONPATH` incluye `/opt/proyecto/src`.
- Lo que depende de la máquina va en `airflow/.env`, que no se versiona; la
  plantilla es `airflow/.env.example`.
- Cada paso del pipeline es una tarea del DAG y recibe `fecha = {{ ds }}`.
- `dags/prediccion_diaria.py`: diario a las 06:00 UTC del 2012-01-01 al 2012-01-31
  (`catchup=True`), cinco tareas en cadena que se pasan por XCom la ruta escrita. Solo
  orquesta; la lógica está en `src/bicis/`.
- Backfill (por ejemplo, de enero con 4 corridas a la vez): `airflow backfill create --dag-id
  prediccion_diaria --from-date 2012-01-01 --to-date 2012-01-31T23:59:59
  --max-active-runs 4`. El `--to-date` tiene que llegar hasta el final del día, porque
  la corrida es a las 06:00.

## MLflow

- `docker/docker-compose.yml`: servicio `mlflow` (`ghcr.io/mlflow/mlflow:v3.17.2`) en
  http://localhost:5000. Usa SQLite (`/mlflow/mlflow.db`) y artefactos en
  `/mlflow/artifacts`, los dos en el volumen `demanda-bicis-c5_mlflow-data`. El servidor
  recibe y guarda los artefactos (`--serve-artifacts`), así que los clientes solo usan la URL.
- Ese compose crea la red `bicis-net`. Airflow se une a ella como red externa y tiene
  `MLFLOW_TRACKING_URI=http://mlflow:5000`. Por eso `make mlflow-up` va antes que
  Airflow (`make airflow-up` ya lo llama).
- `--allowed-hosts` acepta solo `localhost`, `127.0.0.1` y `mlflow` (con y sin `:5000`).
  Otro nombre de host recibe un rechazo.

## Dependencias y comandos

- `requirements.txt`: dependencias del pipeline, con versiones fijas y compatibles
  con las restricciones de Airflow 3.3.2 (Python 3.12). `requirements-dev.txt` agrega
  `dvc` y `pytest` para el `.venv`.
- `make venv` · `make fuente` · `make test` · `make entrenar` ·
  `make evaluar RUN_ID=... MES=...` · `make pipeline FECHA=AAAA-MM-DD` · `make mlflow-up` · `make airflow-build` ·
  `make airflow-up` · `make airflow-down`

## Pruebas

- pytest con datos sintéticos en `tmp_path` (fixtures `entorno` y `modelo_entrenado` en
  `tests/conftest.py`; `pyproject.toml` agrega `src/` al path). Las pruebas no dependen de que exista
  `fuente/bicis.db` ni el modelo real. Para apuntar a otras rutas, usan las variables
  `BICIS_*`.
