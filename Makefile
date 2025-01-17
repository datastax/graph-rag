# Variables
RUFF = ruff@0.8.6
DEPTRY = deptry@0.22.0

.PHONY: check
check:
	@echo "🚀 Linting code: Running ruff check"
	@uvx $(RUFF) check .

.PHONY: fix
fix:
	@echo "🚀 Linting code: Running ruff check --fix"
	@uvx $(RUFF) check --fix .

.PHONY: fmt
fmt:
	uvx $(RUFF) format .

.PHONY: fmt-check
fmt-check:
	uvx $(RUFF) format --check .

.PHONY: check-lock
check-lock:
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked

.PHONY: deptry
deptry:
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@uvx $(DEPTRY) packages/langchain-graph-rag/src packages/langchain-graph-rag/tests

.PHONY: docker-up
docker-up:
	docker compose up -d
	./scripts/healthcheck.sh

.PHONY: docker-down
docker-down:
	docker compose down --rmi local

.PHONY: sync-langchain-graph-rag
sync-langchain-graph-rag:
	@uv sync --package langchain-graph-rag

.PHONY: sync-graph-rag
sync-graph-rag:
	@uv sync --package graph-rag

.PHONY: integration
integration:
	@echo "🚀 Testing code: Running pytest ./packages/langchain-graph-rag/tests/integration_tests (in memory only)"
	@uv run --project langchain-graph-rag pytest -vs ./packages/langchain-graph-rag/tests/integration_tests/

.PHONY: unit
unit:
	@echo "🚀 Testing code: Running pytest ./packages/langchain-graph-rag/tests/unit_tests/"
	@uv run --project langchain-graph-rag pytest -vs ./packages/langchain-graph-rag/tests/unit_tests/

.PHONY: test
test: sync-langchain-graph-rag
	@echo "🚀 Testing code: Running pytest"
	@uv run --project langchain-graph-rag python -m pytest -vs ./packages/langchain-graph-rag/tests/ --stores=all

.PHONY: mypy
mypy:
	@echo "🚀 Static type checking: Running mypy"
	@uv run --project langchain-graph-rag mypy ./packages/langchain-graph-rag

lint: fmt fix mypy

.PHONY: build-langchain-graph-rag
build-langchain-graph-rag: sync-langchain-graph-rag
	@echo "🚀 Building langchain-graph-rag package"
	@uv build --package langchain-graph-rag

.PHONY: build-graph-rag
build-graph-rag: sync-graph-rag
	@echo "🚀 Building graph-rag package"
	@uv build --package graph-rag
