"""Unit tests for tests.helpers.utils.build_test_xml.

Verifies that build_test_xml produces a well-formed XML file that references
the correct paths and fit parameters, without running BASTA.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

from tests.helpers.utils import build_test_xml

_DATA_DIR = Path(__file__).parent.parent / "data"
_TEST_GRID_PATH = _DATA_DIR / "test_grid.hdf5"
_TEST_ASCII_PATH = _DATA_DIR / "test_observables.ascii"


def test_build_test_xml_creates_file(tmp_path):
    """build_test_xml writes a file at the given xml_path."""
    xml_path = tmp_path / "test_input.xml"
    build_test_xml(
        grid_path=_TEST_GRID_PATH,
        ascii_path=_TEST_ASCII_PATH,
        output_path=tmp_path,
        xml_path=xml_path,
    )
    assert xml_path.exists()
    assert xml_path.stat().st_size > 0


def test_build_test_xml_returns_the_xml_path(tmp_path):
    """build_test_xml returns the path it wrote to."""
    xml_path = tmp_path / "test_input.xml"
    returned = build_test_xml(
        grid_path=_TEST_GRID_PATH,
        ascii_path=_TEST_ASCII_PATH,
        output_path=tmp_path,
        xml_path=xml_path,
    )
    assert returned == xml_path


def test_build_test_xml_produces_well_formed_xml(tmp_path):
    """The file produced by build_test_xml is valid, parseable XML."""
    xml_path = tmp_path / "test_input.xml"
    build_test_xml(
        grid_path=_TEST_GRID_PATH,
        ascii_path=_TEST_ASCII_PATH,
        output_path=tmp_path,
        xml_path=xml_path,
    )
    # ET.parse raises ParseError if the XML is malformed
    tree = ET.parse(xml_path)
    assert tree.getroot() is not None


def test_build_test_xml_embeds_grid_path_and_star_data(tmp_path):
    """The produced XML contains the grid path and the star's observable values.

    generate_xml reads the ASCII file and inlines each star's values directly
    into the XML — it does not store the ASCII file path.  The grid path is
    stored as a <library path="..."> attribute.
    """
    xml_path = tmp_path / "test_input.xml"
    build_test_xml(
        grid_path=_TEST_GRID_PATH,
        ascii_path=_TEST_ASCII_PATH,
        output_path=tmp_path,
        xml_path=xml_path,
    )
    xml_text = xml_path.read_text()
    # Grid path is stored in the XML
    assert str(_TEST_GRID_PATH.resolve()) in xml_text
    # Star data from test_observables.ascii is inlined into the XML
    assert 'starid="teststar"' in xml_text
    assert 'value="5825"' in xml_text  # Teff


def test_build_test_xml_configures_teff_logg_feh_as_fit_parameters(tmp_path):
    """The produced XML fits Teff, logg, and FeH."""
    xml_path = tmp_path / "test_input.xml"
    build_test_xml(
        grid_path=_TEST_GRID_PATH,
        ascii_path=_TEST_ASCII_PATH,
        output_path=tmp_path,
        xml_path=xml_path,
    )
    xml_text = xml_path.read_text()
    assert "Teff" in xml_text
    assert "logg" in xml_text
    assert "FeH" in xml_text
