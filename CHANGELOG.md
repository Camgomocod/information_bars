# Changelog

All notable public changes to `trading-core` are documented here.

## [0.1.0] - Unreleased

### Added

- Installable Python package with CLI entry points.
- Offline healthcheck and deterministic synthetic DRB example.
- Optional dependency groups for database, GPU, visualization, notebooks, and development.
- Contributor, security, code-of-conduct, data-terms, and citation documentation.
- CI smoke test for installation and the public synthetic workflow.

### Changed

- Docker image builds from `pyproject.toml` and includes a healthcheck.
- Configuration can be selected with `TRADING_CORE_CONFIG_DIR`.
- Database-dependent tests are explicitly marked and skipped by the non-DB suite.

### Notes

- Historical exchange data and Optuna artifacts are not part of the software release.
- The MIT license covers the original source code, not automatically exchange-derived datasets.
