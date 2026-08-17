"""Console entry points for the supported user-facing commands."""


def healthcheck() -> int:
    """Run the installation healthcheck without relying on the current directory."""
    from src.healthcheck import main

    return main()


def synthetic_example() -> int:
    """Run the dependency-free-from-network synthetic pipeline example."""
    from src.examples.synthetic_pipeline import main

    return main()
