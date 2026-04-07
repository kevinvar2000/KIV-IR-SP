"""Application entry point that launches interactive mode."""

from interactive import interactive_mode


def main() -> int:
    """Run the interactive application and return its exit code."""
    return interactive_mode()


if __name__ == "__main__":
    raise SystemExit(main())
