def pytest_addoption(parser):
    parser.addoption(
        "--overwrite-results", action="store_true", default=False, help="Overwrite test results"
    )
