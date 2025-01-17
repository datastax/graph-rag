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
	@uvx $(DEPTRY) src tests

.PHONY: docker-up
docker-up:
	docker compose up -d
	./scripts/healthcheck.sh

.PHONY: docker-down
docker-down:
	docker compose down --rmi local

.PHONY: integration
integration:
	@echo "🚀 Testing code: Running pytest ./tests/inegration_tests (in memory only)"
	@uv run pytest -vs ./tests/integration_tests/

.PHONY: unit
unit:
	@echo "🚀 Testing code: Running pytest ./tests/unit_tests"
	@uv run pytest -vs ./tests/unit_tests/

.PHONY: test
test:
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest -vs ./tests/unit_tests ./tests/integration_tests/ --stores=all

.PHONY: mypy
mypy:
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy .

lint: fmt fix mypy
