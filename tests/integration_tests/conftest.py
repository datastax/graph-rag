from pytest import Parser


def pytest_addoption(parser: Parser):
    parser.addoption(
        "--in-memory-only",
        action="store_true",
        default=False,
        help="Run integration tests with only the in-memory database types.",
    )
