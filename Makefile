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

.PHONY: check-lock lock
check-lock:
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked

lock:
	@echo "🚀 Updating lock files"
	@uv lock

.PHONY: deptry-gr deptry-lgr deptry
deptry-gr:
	@echo "🚀 Checking for obsolete dependencies: Running deptry on graph-retriever"
	cd packages/graph-retriever && uvx $(DEPTRY) src tests

deptry-lgr:
	@echo "🚀 Checking for obsolete dependencies: Running deptry on langchain-graph-retriever"
	cd packages/langchain-graph-retriever && uvx $(DEPTRY) src tests

deptry: deptry-gr deptry-lgr

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
integration:
	@echo "🚀 Testing code: Running pytest ./packages/langchain-graph-retriever/tests/integration_tests (in memory only)"
	@uv run --package langchain-graph-retriever pytest -vs ./packages/langchain-graph-retriever/tests/integration_tests/

.PHONY: unit
unit:
	@echo "🚀 Testing code: Running pytest ./packages/langchain-graph-retriever/tests/unit_tests/"
	@uv run --package langchain-graph-retriever pytest -vs ./packages/langchain-graph-retriever/tests/unit_tests/

.PHONY: test
test: sync-langchain-graph-retriever
	@echo "🚀 Testing code: Running pytest"
	@cd packages/langchain-graph-retriever && uv run pytest -vs . --stores=all

.PHONY: mypy
mypy:
	@echo "🚀 Static type checking: Running mypy"
	@uv run --package langchain-graph-retriever mypy ./packages/langchain-graph-retriever

lint: fmt fix mypy

.PHONY: build-langchain-graph-retriever
build-langchain-graph-retriever: sync-langchain-graph-retriever
	@echo "🚀 Building langchain-graph-retriever package"
	@uv build --package langchain-graph-retriever

.PHONY: build-graph-retriever
build-graph-retriever: sync-graph-retriever
	@echo "🚀 Building graph-retriever package"
	@uv build --package graph-retriever
