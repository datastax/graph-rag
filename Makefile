# Variables
RUFF = ruff@0.8.6
DEPTRY = deptry@0.22.0

.PHONY: check
check: ## Run `ruff check` to fail if lint errors (fix with `make fix`).
	@echo "🚀 Linting code: Running ruff check"
	@uvx $(RUFF) check .

.PHONY: fix
fix: ## Run `ruff check --fix` to fix lint errors.
	@echo "🚀 Linting code: Running ruff check --fix"
	@uvx $(RUFF) check --fix .

.PHONY: fmt
fmt: ## Run `ruff format` to fix formatting errors.
	uvx $(RUFF) format .

.PHONY: fmt-check
fmt-check: ## Run `ruff format --check` to check for errors (fix with `make fmt`).
	uvx $(RUFF) format --check .

.PHONY: check-lock lock
check-lock: ## Run `uv lock --locked` to check consistency (fix with `make lock`).
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked

lock: ## Run `uv lock` to fix consistency.
	@echo "🚀 Updating lock files"
	@uv lock

.PHONY: deptry-gr deptry-lgr deptry
deptry-gr:
	@echo "🚀 Checking for obsolete dependencies: Running deptry on graph-retriever"
	cd packages/graph-retriever && uvx $(DEPTRY) src tests

deptry-lgr:
	@echo "🚀 Checking for obsolete dependencies: Running deptry on langchain-graph-retriever"
	cd packages/langchain-graph-retriever && uvx $(DEPTRY) src tests

deptry: deptry-gr deptry-lgr ## Check for dependency issues.

.PHONY: docker-up
docker-up:
	docker compose up -d
	./scripts/healthcheck.sh

.PHONY: docker-down
docker-down:
	docker compose down --rmi local

.PHONY: sync-langchain-graph-retriever
sync-langchain-graph-retriever:
	@uv sync --package langchain-graph-retriever

.PHONY: sync-graph-retriever
sync-graph-retriever:
	@uv sync --package graph-retriever

.PHONY: integration
integration: ## Run Integration tests (in-memory only).
	@echo "🚀 Testing code: Running pytest ./packages/langchain-graph-retriever/tests/integration_tests (in memory only)"
	@uv run --package langchain-graph-retriever pytest -vs ./packages/langchain-graph-retriever/tests/integration_tests/

.PHONY: unit ## Run unit tests.
unit:
	@echo "🚀 Testing code: Running pytest ./packages/langchain-graph-retriever/tests/unit_tests/"
	@uv run --package langchain-graph-retriever pytest -vs ./packages/langchain-graph-retriever/tests/unit_tests/

.PHONY: test
test: sync-langchain-graph-retriever ## Run all tests (against all stores).
	@echo "🚀 Testing code: Running pytest"
	@cd packages/langchain-graph-retriever && uv run pytest -vs . --stores=all

.PHONY: mypy
mypy: ## Check for mypy errors.
	@echo "🚀 Static type checking: Running mypy"
	@uv run --package langchain-graph-retriever mypy ./packages/langchain-graph-retriever

lint: fmt fix mypy # Run all lints (fixing where possible).

.PHONY: build-langchain-graph-retriever
build-langchain-graph-retriever: sync-langchain-graph-retriever
	@echo "🚀 Building langchain-graph-retriever package"
	@uv build --package langchain-graph-retriever

.PHONY: build-graph-retriever
build-graph-retriever: sync-graph-retriever
	@echo "🚀 Building graph-retriever package"
	@uv build --package graph-retriever

apidocs: ## Update package installation and generate docs
	uv pip install packages/* --force-reinstall
	cd docs && uv run quartodoc build

docs-preview: apidocs ## Live docs
	cd docs && uv run quarto preview

docs-build: apidocs ## Generate docs site in `_site`.
	cd docs && uv run quarto render

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help