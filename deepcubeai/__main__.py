"""Module entry: `python -m deepcubeai` runs the Python pipeline CLI."""

from __future__ import annotations

from . import pipeline


def main() -> None:
    """Entrypoint for `python -m deepcubeai`; forwards to the pipeline CLI."""
    pipeline.main()


if __name__ == "__main__":
    main()
