from cli import parse_args
from interactive import interactive_mode
from pipeline import run_non_interactive


def main() -> int:
    args = parse_args()
    if args.stage == "interactive":
        return interactive_mode()
    return run_non_interactive(args)


if __name__ == "__main__":
	raise SystemExit(main())
