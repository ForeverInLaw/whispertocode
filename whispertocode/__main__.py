from .cli import main as app_main


def main() -> int:
    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
