.DEFAULT_GOAL := help

PYTHON  := .venv/bin/python
PIP     := .venv/bin/pip
COMPOSE := docker compose --project-directory airflow -f airflow/docker-compose.yaml

.PHONY: help venv fuente test airflow-env airflow-build airflow-up airflow-down airflow-ps airflow-logs clean

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

airflow-env: ## Crea airflow/.env para esta máquina (si no existe)
	@test -f airflow/.env || { \
	  sed -e "s|^AIRFLOW_UID=.*|AIRFLOW_UID=$$(id -u)|" \
	      -e "s|^PROYECTO_DIR=.*|PROYECTO_DIR=$(CURDIR)|" \
	      airflow/.env.example > airflow/.env && echo "airflow/.env creado"; }

airflow-build: airflow-env ## Construye la imagen propia de Airflow
	$(COMPOSE) build

airflow-up: airflow-env ## Levanta Airflow en segundo plano (http://localhost:8080)
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
