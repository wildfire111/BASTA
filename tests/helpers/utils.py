import h5py
from pathlib import Path


def try_float(s):
    """Return float(s) if s is numeric, otherwise return s unchanged."""
    try:
        return float(s)
    except ValueError:
        return s


def parse_results_ascii(path):
    """Return {column_name: value} for the first data row of a BASTA results ASCII file.

    The column header is taken from the last comment line (lines starting with '#').
    Numeric values are returned as floats; non-numeric values (e.g. starid) stay as strings.
    """
    lines = Path(path).read_text().splitlines()
    last_comment_line = [line for line in lines if line.startswith("#")][-1]
    column_names = last_comment_line.lstrip("# ").split()
    first_data_row = [line for line in lines if line and not line.startswith("#")][0].split()
    return {col: try_float(val) for col, val in zip(column_names, first_data_row)}


def build_test_xml(grid_path, ascii_path, output_path, xml_path):

    from basta.xml_create import generate_xml

    define_io = {
        "gridfile": str(grid_path.resolve()),
        "outputpath": str(output_path.resolve()),
        "asciifile": str(ascii_path.resolve()),
        "params": (
            "starid",
            "Teff",
            "Teff_err",
            "logg",
            "logg_err",
            "FeH",
            "FeH_err",
        ),
    }

    define_fit = {
        "fitparams": ("Teff", "logg", "FeH"),
        "priors": {
            "Teff": {"sigmacut": "3"},
            "logg": {"sigmacut": "3"},
            "FeH":  {"sigmacut": "3"},
        },
        "solarmodel": False,
    }

    define_output = {
        "outparams": ("Teff", "logg", "FeH"),
        "outputfile": "test_results.ascii",
        "optionaloutputs": False,
    }

    define_plots = {
        "cornerplots": (),
        "kielplots": False,
        "freqplots": False,
    }

    define_intpol = {}

    xmldata = generate_xml(
        **define_io,
        **define_fit,
        **define_output,
        **define_plots,
        **define_intpol,
    )

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xmldata)

    print("Test XML created successfully at:", xml_path)

    return xml_path


def build_test_grid_from_garstec(source_path: Path, destination_path: Path):
    """Extract a single-track test grid from the full Garstec HDF5 grid.

    Reads ``source_path`` (the full Garstec_16CygA.hdf5 grid) and writes a
    minimal single-track HDF5 file to ``destination_path``.  The output file
    preserves the same group structure as the source (grid/tracks, header,
    solar_models) so that BASTA's normal I/O routines can read it without
    modification.

    Parameters
    ----------
    source_path:
        Absolute path to the full Garstec HDF5 grid that ships with the BASTA
        examples.
    destination_path:
        Absolute path where the reduced test grid will be written.  The parent
        directory must already exist (or be created by the caller).
    """

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source grid not found: {source_path}\n"
            "Please check the installation guide to obtain the necessary data files."
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_path, "r") as source_grid, \
         h5py.File(destination_path, "w") as destination_grid:

        target_track_name = "track001"
        target_track_index = 0

        # Build the same group hierarchy as the source grid
        grid_group = destination_grid.create_group("grid")
        tracks_group = grid_group.create_group("tracks")

        # Copy only the first track so the test grid stays small
        source_grid.copy(f"grid/tracks/{target_track_name}", tracks_group)

        destination_header = destination_grid.create_group("header")
        source_header = source_grid["header"]

        # Copy header datasets, slicing array datasets to the single track row
        for dataset_name, source_dataset in source_header.items():

            if source_dataset.shape == ():
                # Scalar dataset — copy verbatim
                dataset_data = source_dataset[()]
            else:
                # Array dataset — keep only the row for the target track
                dataset_data = source_dataset[
                    target_track_index : target_track_index + 1
                ]

            destination_header.create_dataset(
                dataset_name,
                data=dataset_data,
                dtype=source_dataset.dtype,
            )

        destination_solar_models = destination_grid.create_group("solar_models")
        # Solar models are grid-wide, not per-track, so copy the whole group
        source_grid.copy("solar_models", destination_solar_models)

    print(f"Test grid written to: {destination_path}")
