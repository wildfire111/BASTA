import h5py
import numpy as np
from pathlib import Path


def build_test_grid_from_garstec():

    #Build relative paths to the source and destination files
    PROJECT_ROOT = Path(__file__).parent.parent
    SOURCE_DIR = PROJECT_ROOT / "examples" / "grids"
    DESTINATION_DIR = PROJECT_ROOT / "tests" / "data"
    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)

    source_path = SOURCE_DIR / "Garstec_16CygA.hdf5"

    if not source_path.exists():
        raise FileNotFoundError(f"File {source_path} does not exist. Please check the installation guide to obtain the necessary data files.")

    with h5py.File(source_path, "r") as src, \
        h5py.File(DESTINATION_DIR / "test_grid.hdf5", "w") as dst:

        target_track = "track001"
        track_index = 0

        #Build the grid structure in the destination file identical to the source file
        grid_group = dst.create_group("grid")
        tracks_group = grid_group.create_group("tracks")

        #Copy just one track from the source file to the destination file
        src.copy(f"grid/tracks/{target_track}", tracks_group)

        header_dst = dst.create_group("header")
        header_src = src["header"]

        #Copy the header groups appropriate for the target track.
        for name, dset in header_src.items():
            
            if dset.shape == (): #dset is scalar, copy as is
                data = dset[()]
            else: #dset is an array, copy only the relevant slice for the target track
                data = dset[track_index:track_index+1]

            header_dst.create_dataset(
                name,
                data=data,
                dtype=dset.dtype
            )

        solar_models_dst = dst.create_group("solar_models")
        #Copy the whole solar_models group
        src.copy("solar_models", solar_models_dst)

        print("Test grid created successfully at:", DESTINATION_DIR / "test_grid.hdf5")



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
        ),
    }

    define_fit = {
        "fitparams": ("Teff",),
        "priors": {"Teff": {"sigmacut": "3"}},
        "solarmodel": False,
    }

    define_output = {
        "outparams": ("Teff",),
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

if __name__ == "__main__":
    build_test_grid_from_garstec()
    data_dir = Path(__file__).parent / "data"
    ascii_path = data_dir / "test_observables.ascii"
    build_test_xml(
        grid_path=data_dir / "test_grid.hdf5",
        ascii_path=ascii_path,
        output_path=data_dir,
        xml_path=data_dir / "test_input.xml"
    )

        


