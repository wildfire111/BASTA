from pathlib import Path
from helpers import build_test_xml
import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def test_grid_path():
    return Path(__file__).parent / "data" / "test_grid.hdf5"

@pytest.fixture(scope="session")
def test_run_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("basta_run")

@pytest.fixture(scope="session")
def test_ascii_path():
    return Path(__file__).parent / "data" / "test_observables.ascii"

@pytest.fixture(scope="session")
def test_xml_path(test_grid_path, test_ascii_path, test_run_dir):
    xml_path = test_run_dir / "test_input.xml"

    build_test_xml(
        grid_path=test_grid_path,
        ascii_path=test_ascii_path,
        output_path=test_run_dir,
        xml_path=xml_path,
    )

    return xml_path
