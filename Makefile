.PHONY: dev test lint build seed format

dev: ## Start API server + dashboard dev server
	@echo "Starting API server and dashboard..."
	uvicorn server.main:app --reload --port 8000 & \
	cd dashboard && pnpm dev

test: ## Run all tests
	pytest tests/ -v --tb=short

lint: ## Run ruff linter
	ruff check ml/ server/ tests/

format: ## Auto-format with ruff
	ruff format ml/ server/ tests/
	ruff check --fix ml/ server/ tests/

build: ## Build dashboard + Docker image
	cd dashboard && pnpm install && pnpm build
	docker build -t netra-ai:latest .

seed: ## Seed database
	python -m scripts.seed
