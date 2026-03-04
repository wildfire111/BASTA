import dataclasses
from pathlib import Path


class BastaHarness:
    """
    Calls bastamain.BASTA() directly and uses spies to capture outputs.
    Accepts Grid + Star models; bypasses the XML layer entirely.

    Currently runs with usebayw=False (Bayesian weights disabled). To enable weights,
    Track.values must include a dage array and the Grid must expose active_weights in
    its header. See util.read_grid_bayweights() and bastamain.py for details.
    """

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.selectedmodels = None
        self._install_spies()

    def _install_spies(self):
        import basta.bastamain as bastamain

        actual = bastamain.stats.get_highest_likelihood

        def fake(Grid, selectedmodels, inputparams):
            self.selectedmodels = selectedmodels
            return actual(Grid, selectedmodels, inputparams)

        self.monkeypatch.setattr(bastamain.stats, "get_highest_likelihood", fake)

    def _build_fitparams(self, star):
        """Extract {param: (value, error)} pairs from a Star's dataclass fields."""
        fitparams = {}
        for f in dataclasses.fields(star):
            if f.name.endswith("_err"):
                continue
            val = getattr(star, f.name)
            err = getattr(star, f"{f.name}_err")
            if val is not None and err is not None:
                fitparams[f.name] = (val, err)
        return fitparams

    def _build_inputparams(self, fitparams, output_dir, fout, ferr):
        """
        Build the inputparams dict for bastamain.BASTA().

        BASTA takes a large flat dict. Keys fall into three groups:
          - Derived from Star: fitparams, asciiparams
          - Per-run infrastructure: output path and file handles
          - Harness defaults: features that are off for test simplicity
            (no plots, no frequency fitting)
        """
        return {
            # Derived from Star
            "fitparams": fitparams,
            "asciiparams": list(fitparams.keys()),

            # Per-run infrastructure
            "output": str(output_dir) + "/",  # trailing slash required by BASTA
            "asciioutput": fout,
            "erroutput": ferr,
            "warnoutput": False,

            # Harness defaults — features disabled for test simplicity
            "cornerplots": [],
            "kielplots": [],              # must be list, not bool (len() is called on it)
            "freqplots": False,
            "fitfreqs": {"active": False},
            "limits": {},
            "magnitudes": {},
            "centroid": "median",
            "uncert": "quantiles",
            "plotfmt": "png",
            "nameinplot": False,
            "solarmodel": "False",        # str not bool: BASTA calls strtobool() on this
        }

    def run(self, grid, star, tmp_path, overrides=None, usepriors=(), usebayw=False):
        """
        Run BASTA directly using Grid and Star models.

        Parameters
        ----------
        grid : tests.helpers.models.grid.Grid
        star : tests.helpers.models.star.Star
        tmp_path : Path  (pytest tmp_path fixture)
        overrides : dict, optional
            Override any harness-default inputparams key (e.g. {"centroid": "mean"}).
        """
        from basta.bastamain import BASTA

        grid_path = Path(tmp_path) / "harness_grid.hdf5"
        grid.to_hdf5(grid_path)

        fitparams = self._build_fitparams(star)

        output_dir = Path(tmp_path) / "output"
        output_dir.mkdir()

        with (
            open(output_dir / "results.ascii", "ab+") as fout,
            open(output_dir / "results.err", "a+") as ferr,
        ):
            inputparams = self._build_inputparams(fitparams, output_dir, fout, ferr)
            if overrides:
                inputparams.update(overrides)
            BASTA(
                starid="test",
                gridfile=str(grid_path),
                inputparams=inputparams,
                usebayw=usebayw,
                usepriors=usepriors,
            )

    @property
    def best_fit_model(self):
        """Return (maxPDF_path, maxPDF_ind), or None if BASTA has not run."""
        if self.selectedmodels is None:
            return None
        from basta.stats import most_likely

        return most_likely(self.selectedmodels)
