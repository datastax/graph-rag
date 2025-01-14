# Variables
RUFF = ruff@0.8.6


.PHONY: check
check:
	uvx $(RUFF) check .

.PHONY: fix
fix:
	uvx $(RUFF) check . --fix

.PHONY: fmt
fmt:
	uvx $(RUFF) format .

.PHONY: docker-up
docker-up:
	docker compose up -d
	./scripts/healthcheck.sh

.PHONY: docker-down
docker-down:
	docker compose down --rmi local

.PHONY: integration
integration:
	uv run pytest -vs ./tests/integration_tests/

.PHONY: in-memory
in-memory:
	uv run pytest -vs --in-memory-only ./tests/integration_tests/

.PHONY: unit
unit:
	uv run pytest -vs ./tests/unit_tests/

.PHONY: mypy
mypy:
	uv run mypy .

lint: fmt fix mypy
