lint:
	uv run ruff check
	uv run ruff format --check .
.PHONY: lint

fix:
	uv run ruff check --fix
	uv run ruff format .
.PHONY: fix
