"""Unit tests for tests.helpers.utils — the shared test utility functions.

These tests cover try_float and parse_results_ascii directly, without running
BASTA or touching any real results files.
"""
from pathlib import Path

from tests.helpers.utils import parse_results_ascii, try_float


# ---------------------------------------------------------------------------
# try_float
# ---------------------------------------------------------------------------

def test_try_float_converts_valid_string_to_float():
    """A string that represents a valid number is returned as a float."""
    result = try_float("3.14")
    assert result == 3.14
    assert isinstance(result, float)


def test_try_float_returns_string_for_non_numeric():
    """A string that cannot be converted to float is returned unchanged."""
    result = try_float("teststar")
    assert result == "teststar"
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# parse_results_ascii
# ---------------------------------------------------------------------------

def test_parse_results_ascii_returns_dict_of_column_values(tmp_path):
    """parse_results_ascii returns a dict mapping column names to row values."""
    results_file = tmp_path / "results.ascii"
    results_file.write_text(
        "# starid Teff\n"
        "teststar 5800.0\n"
    )
    result = parse_results_ascii(results_file)
    assert result == {"starid": "teststar", "Teff": 5800.0}


def test_parse_results_ascii_returns_float_for_numeric_columns(tmp_path):
    """Numeric column values are returned as floats, not strings."""
    results_file = tmp_path / "results.ascii"
    results_file.write_text(
        "# starid Teff logg\n"
        "star1 5750.5 4.44\n"
    )
    result = parse_results_ascii(results_file)
    assert isinstance(result["Teff"], float)
    assert isinstance(result["logg"], float)
    assert result["Teff"] == 5750.5
    assert result["logg"] == 4.44


def test_parse_results_ascii_returns_string_for_starid_column(tmp_path):
    """The starid column value stays as a string because it is non-numeric."""
    results_file = tmp_path / "results.ascii"
    results_file.write_text(
        "# starid Teff\n"
        "16CygA 5825.0\n"
    )
    result = parse_results_ascii(results_file)
    assert isinstance(result["starid"], str)
    assert result["starid"] == "16CygA"


def test_parse_results_ascii_uses_last_hash_line_as_header(tmp_path):
    """When the file has multiple comment lines, the last one is used as the column header.

    BASTA writes a multi-line preamble before the column header, so this
    behaviour is required for real output files.
    """
    results_file = tmp_path / "results.ascii"
    results_file.write_text(
        "# BASTA v1.5.3 output\n"
        "# Generated: 2026-01-01\n"
        "# starid Teff logg\n"
        "mystar 5900.0 4.20\n"
    )
    result = parse_results_ascii(results_file)
    assert set(result.keys()) == {"starid", "Teff", "logg"}
    assert result["starid"] == "mystar"
    assert result["Teff"] == 5900.0
    assert result["logg"] == 4.20


def test_parse_results_ascii_handles_extra_whitespace_in_header(tmp_path):
    """Column names separated by multiple spaces in the header are parsed correctly."""
    results_file = tmp_path / "results.ascii"
    results_file.write_text(
        "#  starid   Teff   Teff_errp\n"
        "teststar  5791.5  11.9\n"
    )
    result = parse_results_ascii(results_file)
    assert set(result.keys()) == {"starid", "Teff", "Teff_errp"}
    assert result["starid"] == "teststar"
    assert result["Teff"] == 5791.5
    assert result["Teff_errp"] == 11.9
