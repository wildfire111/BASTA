from pathlib import Path
from tests.helpers.utils import build_test_grid_from_garstec, build_test_xml
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden", action="store_true", default=False,
        help="Regenerate golden files from current BASTA output.",
    )
    parser.addoption(
        "--regenerate-test-grid", action="store_true", default=False,
        help="Regenerate test_grid.hdf5 from the full Garstec example grid.",
    )


@pytest.fixture(scope="session", autouse=True)
def regenerate_test_grid_if_requested(request, project_root):
    if not request.config.getoption("--regenerate-test-grid"):
        return

    source_path = project_root / "examples" / "grids" / "Garstec_16CygA.hdf5"
    destination_path = project_root / "tests" / "data" / "test_grid.hdf5"

    build_test_grid_from_garstec(source_path, destination_path)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def test_grid_path():
    return Path(__file__).parent / "data" / "test_grid.hdf5"

@pytest.fixture(scope="session")
def test_ascii_path():
    return Path(__file__).parent / "data" / "test_observables.ascii"

@pytest.fixture
def test_xml_path(test_grid_path, test_ascii_path, tmp_path):
    """Build an XML config in a per-test temporary directory.

    Function-scoped so that each test that runs BASTA gets its own isolated
    output directory and cannot be affected by another test's output files.
    """
    xml_path = tmp_path / "test_input.xml"

    build_test_xml(
        grid_path=test_grid_path,
        ascii_path=test_ascii_path,
        output_path=tmp_path,
        xml_path=xml_path,
    )

    return xml_path
