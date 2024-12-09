lint:
	uv run --index-strategy unsafe-best-match -- ruff check
	uv run --index-strategy unsafe-best-match -- ruff format --check .
.PHONY: lint

fix:
	uv run --index-strategy unsafe-best-match -- ruff check --fix
	uv run --index-strategy unsafe-best-match -- ruff format .
.PHONY: fix
