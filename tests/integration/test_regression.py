"""Regression tests for the full BASTA pipeline.

These tests run the real grid (tests/data/test_grid.hdf5) through the
complete BASTA pipeline — XML parsing → Bayesian fitting → ASCII output —
and compare the results against a checked-in golden file.

Golden file: tests/data/expected_results.ascii
Regenerate:  pytest --update-golden
"""
from pathlib import Path
import shutil

import pytest

from tests.helpers.utils import parse_results_ascii

_DATA_DIR = Path(__file__).parent.parent / "data"
_GOLDEN_RESULTS = _DATA_DIR / "expected_results.ascii"


@pytest.mark.regression
def test_xml_end_to_end(test_xml_path, request):
    """An end-to-end test to check that the entire pipeline runs and produces expected output."""
    from basta.run import run_xml

    # Fix the numpy random seed before calling BASTA so sampling in
    # process_output.py (np.random.choice) produces deterministic results.
    import numpy as np
    np.random.seed(42)
    run_xml(str(test_xml_path), seed=42)

    run_dir = test_xml_path.parent
    results_file = run_dir / "test_results.ascii"
    xml_output_file = run_dir / "test_results.xml"

    # 1. ASCII results file exists
    assert results_file.exists()

    # 2. ASCII file is not empty
    assert results_file.stat().st_size > 0

    # 3. XML output file exists (ascii_to_xml conversion step)
    assert xml_output_file.exists()

    # 4. XML output is not empty
    assert xml_output_file.read_text().strip()

    # Golden file comparison
    if request.config.getoption("--update-golden"):
        shutil.copy(results_file, _GOLDEN_RESULTS)
        return

    assert _GOLDEN_RESULTS.exists(), (
        "Golden file missing. Run `pytest --update-golden` to generate it."
    )
    actual = parse_results_ascii(results_file)
    expected = parse_results_ascii(_GOLDEN_RESULTS)

    for col, expected_value in expected.items():
        if isinstance(expected_value, float):
            assert actual[col] == pytest.approx(expected_value, rel=1e-5), (
                f"Column '{col}': got {actual[col]!r}, expected {expected_value!r}"
            )


@pytest.mark.regression
def test_cli_runs_without_error(test_xml_path):
    """A simple smoke test to ensure the CLI runs without errors."""
    import subprocess

    result = subprocess.run(
        ["BASTArun", str(test_xml_path)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"CLI failed with error: {result.stderr}"
