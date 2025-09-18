import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--show-plots",
        action="store_true",
        default=False,
        help="Show plots during tests"
    )

@pytest.fixture
def show_plots(request):
    return request.config.getoption("--show-plots")