from __future__ import annotations

from . import pipeline


def main() -> None:
    """Main function to run the DeepCubeAI pipeline."""
    # Delegate to the internal pipeline module
    pipeline.main()


if __name__ == "__main__":
    main()
