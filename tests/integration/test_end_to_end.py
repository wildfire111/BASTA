import pytest



@pytest.mark.regression
def test_cli_smoke_test(test_xml_path):
    """A simple smoke test to ensure the CLI runs without errors."""
    import subprocess

    result = subprocess.run(
        ["BASTArun", str(test_xml_path)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"CLI failed with error: {result.stderr}"

@pytest.mark.regression
def test_end_to_end(test_xml_path):
    """An end-to-end test to check that the entire pipeline runs and produces expected output."""
    from basta.run import run_xml
    
    # Run BASTA using the XML file
    run_xml(str(test_xml_path))

    run_dir = test_xml_path.parent
    results_file = run_dir / "test_results.ascii"
    xml_output_file = run_dir / "test_results.xml"

    # 1. ASCII results file exists
    assert results_file.exists()

    # 2. ASCII file is not empty
    content = results_file.read_text()
    assert content.strip()

    # 3. XML output file exists (ascii_to_xml conversion step)
    assert xml_output_file.exists()

    # 4. XML output is not empty
    assert xml_output_file.read_text().strip()