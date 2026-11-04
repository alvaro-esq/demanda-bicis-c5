.DEFAULT_GOAL := help

PYTHON  := .venv/bin/python
PIP     := .venv/bin/pip
COMPOSE := docker compose --project-directory airflow -f airflow/docker-compose.yaml
MLFLOW  := docker compose -f docker/docker-compose.yml

# Servidor de MLflow (docker/docker-compose.yml). Se puede sobrescribir desde el entorno.
export MLFLOW_TRACKING_URI ?= http://localhost:5000

.PHONY: help venv fuente test entrenar evaluar pipeline mlflow-up mlflow-down airflow-env airflow-build airflow-up airflow-down airflow-ps airflow-logs clean

help: ## Lista los comandos disponibles
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-15s %s\n", $$1, $$2}'

venv: .venv/.instalado ## Crea .venv e instala requirements-dev.txt

.venv/.instalado: requirements.txt requirements-dev.txt
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	touch $@

fuente: venv ## Crea fuente/bicis.db a partir de hour.csv del zip
	$(PYTHON) scripts/crear_fuente.py

test: venv ## Corre las pruebas
	$(PYTHON) -m pytest -q

entrenar: venv ## Entrena (ene-nov 2011), valida (dic 2011) y registra el run en MLflow. MODELO=tipo opcional
	PYTHONPATH=src $(PYTHON) -m bicis.entrenar $(if $(MODELO),--modelo $(MODELO))

evaluar: venv ## Evalúa el modelo del run RUN_ID en MES=AAAA-MM y lo registra en ese run
	@test -n "$(RUN_ID)" -a -n "$(MES)" || { echo "Uso: make evaluar RUN_ID=<id> MES=AAAA-MM"; exit 1; }
	PYTHONPATH=src $(PYTHON) -m bicis.evaluar --run-id $(RUN_ID) --mes $(MES)

pipeline: venv ## Corre los cinco pasos para FECHA=AAAA-MM-DD
	@test -n "$(FECHA)" || { echo "Uso: make pipeline FECHA=AAAA-MM-DD"; exit 1; }
	PYTHONPATH=src $(PYTHON) -c "from bicis import extraer, limpiar, transformar, predecir, almacenar; [print(m.__name__, '->', m.ejecutar('$(FECHA)')) for m in (extraer, limpiar, transformar, predecir, almacenar)]"

mlflow-up: ## Levanta MLflow en http://localhost:5000 y crea la red bicis-net
	$(MLFLOW) up -d

mlflow-down: ## Detiene MLflow (conserva el volumen con runs y artefactos)
	$(MLFLOW) down

airflow-env: ## Crea airflow/.env para esta máquina (si no existe)
	@test -f airflow/.env || { \
	  sed -e "s|^AIRFLOW_UID=.*|AIRFLOW_UID=$$(id -u)|" \
	      -e "s|^PROYECTO_DIR=.*|PROYECTO_DIR=$(CURDIR)|" \
	      airflow/.env.example > airflow/.env && echo "airflow/.env creado"; }

airflow-build: airflow-env ## Construye la imagen propia de Airflow
	$(COMPOSE) build

airflow-up: airflow-env mlflow-up ## Levanta MLflow y Airflow (http://localhost:8080)
	mkdir -p airflow/logs airflow/config airflow/plugins
	$(COMPOSE) up -d

airflow-down: ## Detiene Airflow
	$(COMPOSE) down

airflow-ps: ## Estado de los contenedores
	$(COMPOSE) ps

airflow-logs: ## Sigue los logs de Airflow
	$(COMPOSE) logs -f --tail=100

clean: ## Borra staging y cachés de Python
	rm -rf data/staging .pytest_cache
	find . -name __pycache__ -type d -prune -not -path './.venv/*' -exec rm -rf {} +
