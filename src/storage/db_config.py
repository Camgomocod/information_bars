"""
db_config.py — Database connection config from environment variables.

Never hardcode credentials. Resolution order:
1. ``TRADING_CORE_DATABASE_URL`` — full URL override.
2. Individual ``DB_HOST`` / ``DB_PORT`` / ``DB_NAME`` / ``DB_USER`` / ``DB_PASSWORD``.
3. Local defaults (localhost, trading_core) for a stock ``docker-compose up -d``.
"""

from __future__ import annotations

import os
from urllib.parse import quote_plus

DEFAULT_DB_HOST = "localhost"
DEFAULT_DB_PORT = "5432"
DEFAULT_DB_NAME = "trading_core"
DEFAULT_DB_USER = "trading"
DEFAULT_DB_PASSWORD = "trading"


def get_db_url() -> str:
    url = os.environ.get("TRADING_CORE_DATABASE_URL")
    if url:
        return url

    host = os.environ.get("DB_HOST", DEFAULT_DB_HOST)
    port = os.environ.get("DB_PORT", DEFAULT_DB_PORT)
    name = os.environ.get("DB_NAME", DEFAULT_DB_NAME)
    user = os.environ.get("DB_USER", DEFAULT_DB_USER)
    password = os.environ.get("DB_PASSWORD", DEFAULT_DB_PASSWORD)

    return f"postgresql://{user}:{quote_plus(password)}@{host}:{port}/{name}"
