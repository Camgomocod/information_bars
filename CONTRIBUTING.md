# Contributing

Thanks for improving `trading-core`.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
trading-core-healthcheck --offline
```

Run the fast test suite before opening a pull request:

```bash
python -m pytest tests -m "not db and not slow" -q
ruff check src tests scripts
ruff format --check src tests scripts
```

## Pull Requests

- Keep changes focused and explain the motivation.
- Add or update tests for behavior changes.
- Do not commit raw market data, credentials, database dumps, Optuna studies,
  generated Parquet files, or local logs.
- Preserve the temporal contract: no lookahead bias and no silent tick loss.
- Document changes that affect the dataset schema or feature definitions.

## Data and Experiments

Use synthetic fixtures for tests whenever possible. Exchange downloads,
TimescaleDB, GPU optimization, and large experiments belong to local or
external artifact workflows, not pull requests.
