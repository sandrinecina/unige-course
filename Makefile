.PHONY: help install run clean test lint format venv activate

PYTHON := python3
VENV := .venv
REQUIREMENTS := requirements.txt
UV := uv

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv: ## Create virtual environment
	$(UV) venv $(VENV)
	@echo "Virtual environment created. Run 'make activate' to see activation instructions."

activate: ## Show activation command for the virtual environment
	@echo "To activate the virtual environment, run:"
	@echo "  source $(VENV)/bin/activate"

install: venv ## Install dependencies
	$(UV) pip install -r $(REQUIREMENTS) --python $(VENV)/bin/python
	@echo "Dependencies installed successfully!"

run: ## Run the application with Streamlit
	$(VENV)/bin/streamlit run app.py

run-api: ## Run the FastAPI application
	$(VENV)/bin/uvicorn app:app --reload

test: ## Run tests (configure as needed)
	@echo "No tests configured yet. Add your test command here."

lint: ## Run linting (requires flake8 or ruff)
	@echo "Linting not configured. Install flake8 or ruff and update this target."

format: ## Format code (requires black)
	@echo "Formatting not configured. Install black and update this target."

clean: ## Clean up cache and temporary files
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build

clean-all: clean ## Clean everything including virtual environment
	rm -rf $(VENV)

freeze: ## Update requirements.txt with current packages
	$(UV) pip freeze --python $(VENV)/bin/python > $(REQUIREMENTS)

upgrade: ## Upgrade all packages
	$(UV) pip install --upgrade -r $(REQUIREMENTS) --python $(VENV)/bin/python