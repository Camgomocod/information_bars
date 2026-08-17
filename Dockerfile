# trading-core Dockerfile
# Minimal Python environment for running the pipeline
# Build: docker build -t trading-core .
# Run:   docker run -it trading-core

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata first so Docker can cache dependency installation.
COPY pyproject.toml README.md LICENSE ./

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY optimization/ ./optimization/
COPY config/ ./config/
COPY tests/ ./tests/

RUN pip install --no-cache-dir .

# Create data directories
RUN mkdir -p data/bars/dollar_run data_optimized/training data_optimized/inference logs

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s CMD python scripts/healthcheck.py --offline

# Default command
CMD ["python", "--version"]
