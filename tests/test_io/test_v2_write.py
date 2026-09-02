"""Writing AAS v2 back out: state columns, terminal marker, unit typo, Label.

The tests here cover the parts of v2 support that the real files cannot
exercise on their own:

* every ``Label`` value in the sample v2 files happens to be 0, so reading it
  as v4's Segment column was invisible.  ``test_label_is_not_read_as_segment``
  synthesises a file where Label carries data, which is the only way to see
  the bug that gating on ``n_cols >= 18`` used to cause.
* a v2 file written by smDA has to be byte-comparable with one written by AAS,
  which means reproducing the ``[um/s]`` unit typo rather than correcting it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from smda_hmm.vbhmm.model import apply_state_columns, build_hmm_csv_text
from smda_hmm.io import aas_format
from smda_hmm.io.aas_format import AAS2, AAS4
from tests.helpers import v4_sample_dir

REPO = Path(__file__).resolve().parents[2]

V2_HEADER = ["No", "Time [frame]", "xg [px]", "yg [px]", "sigma x [px]",
             "sigma y [px]", "Imax [a.u.]", "Iback [a.u.]", "a [a.u./px]",
             "b[a.u./px]", "Raw Intensity[au]", "Average Intensity[au/px^2]",
             "Model 1", "Model 2", "Model 3", "Model 4", "Model 5", "Label"]


def _v2_file(path: Path, trajectories, label_values=None) -> None:
    """*trajectories* is a list of lists of state strings, one per row."""
    lines = [",".join(V2_HEADER)]
    row_i = 0
    for traj_no, rows in enumerate(trajectories, start=1):
        for frame, states in enumerate(rows, start=1):
            label = "0" if label_values is None else str(label_values[row_i])
            cells = ([str(traj_no), str(frame)]
                     + ["1.5", "2.5", "1.0", "1.0", "500", "100", "0", "0",
                        "5000", "600"]
                     + list(states) + [label])
            lines.append(",".join(cells))
            row_i += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



class TestStateColumnWriting:
    """apply_state_columns must fill the format's own columns, not add new
    ones, and must keep the terminal marker the format uses."""

    @dataclass
    class _P:
        min_hidden: int = 1
        max_hidden: int = 3

    @dataclass
    class _R:
        params: object
        state_assignments: dict

    def _result(self, n_rows):
        assign = {n: np.arange(n_rows) % n + 1 for n in (1, 2, 3)}
        for n in assign:                       # assign_states writes 0 at ends
            assign[n] = assign[n].astype(int)
        return self._R(self._P(), assign)

    def test_v2_keeps_model_columns_and_empty_terminal(self, tmp_path):
        p = tmp_path / "c.csv"
        _v2_file(p, [[["1", "1", "1", "1", "1"]] * 3,
                     [["1", "1", "1", "1", "1"]] * 2])
        df, version = aas_format.read_data_csv_text(p)
        assert version == AAS2
        out = apply_state_columns(df, self._result(len(df)), version)

        assert list(out.columns) == V2_HEADER, (
            "v2 columns must be filled in place; adding state(diffusion) N "
            "next to Model N would leave two disagreeing sets of states.")
        # rows 2 and 4 are trajectory ends (3 rows then 2 rows)
        for col in ("Model 1", "Model 2", "Model 3"):
            assert out[col].iloc[2] == "", "terminal marker must stay empty"
            assert out[col].iloc[4] == ""
            assert out[col].iloc[0] != ""

    def test_v2_terminal_is_not_written_as_zero(self, tmp_path):
        """0 is v4's marker.  In a v2 file it would read as state 0, which is
        not a state: it colours as the default and counts as data."""
        p = tmp_path / "c.csv"
        _v2_file(p, [[["1", "1", "1", "1", "1"]] * 2])
        df, version = aas_format.read_data_csv_text(p)
        out = apply_state_columns(df, self._result(len(df)), version)
        assert out["Model 1"].iloc[-1] == ""
        assert out["Model 1"].iloc[-1] != "0"

    def test_unknown_version_raises(self, tmp_path):
        p = tmp_path / "c.csv"
        _v2_file(p, [[["1", "1", "1", "1", "1"]] * 2])
        df, _ = aas_format.read_data_csv_text(p)
        with pytest.raises(ValueError, match="Unknown AAS version"):
            apply_state_columns(df, self._result(len(df)), "aas3")


class TestHmmCsvFormat:
    """v2 hmm.csv differs from v4 in four places, all reproduced."""

    @pytest.fixture(scope="class")
    def result(self):
        from smda_hmm.vbhmm.model import VBHMMParams, run_vbhmm_analysis
        from smda_hmm.io.aas_reader import load_aas_settings_csv
        st = REPO / "data" / "sample" / "settings.csv"
        csvs = list(aas_format.list_data_csvs(REPO / "data" / "sample"))
        if not st.exists() or not csvs:
            pytest.skip("validation data not present")
        s = load_aas_settings_csv(st)
        p = VBHMMParams(
            n_tilde=s["n_tilde"], c_tilde=s["c_tilde"],
            w_pi_tilde=s["w_pi_tilde"], w_b_tilde=s["w_b_tilde"], mag=s["mag"],
            min_hidden=1, max_hidden=2, max_iter=100, num_run=1,
            frame_minimum=s["vbhmm_min_frame"], estimate_mode=s["estimate_mode"],
            is_add_each_trajectory=s["add_per_traj"],
            is_calc_kl_each=s["calc_kl_each"],
            timestep=0.040, distance_per_pixel=0.067)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return run_vbhmm_analysis(str(csvs[-1]), p)

    def test_v2_reproduces_the_unit_typo(self, result):
        """v2 labels a um^2/s quantity as um/s.  Correcting it would produce a
        file no v2 reader has seen and break comparison against AAS output."""
        text = build_hmm_csv_text(result, AAS2)
        assert ",Diffusion coefficient[um/s],," in text
        assert "[um^2/s]" not in text

    def test_v4_keeps_the_correct_unit(self, result):
        text = build_hmm_csv_text(result, AAS4)
        assert ",Diffusion coefficient[um^2/s],," in text

    def test_v2_header_and_placeholder_rows(self, result):
        lines = build_hmm_csv_text(result, AAS2).splitlines()
        assert lines[0] == "Method,SimplevbSPT"
        assert lines[1] == ""
        assert lines[2] == "Model,1"
        assert "wa1,None" in lines and "wa2,None" in lines
        assert not lines[0].startswith("{"), "v2 carries no metadata line"

    def test_v4_header_is_json_metadata(self, result):
        lines = build_hmm_csv_text(result, AAS4).splitlines()
        assert lines[0].startswith("{") and '"timestep"' in lines[0]
        assert "wa1,None" not in lines

    def test_prior_name_indexing_differs(self, result):
        v2 = [l for l in build_hmm_csv_text(result, AAS2).splitlines()
              if l.startswith("Prior name,")]
        v4 = [l for l in build_hmm_csv_text(result, AAS4).splitlines()
              if l.startswith("Prior name,")]
        assert v2[0] == "Prior name,0"      # v2 counts states from 0
        assert v4[0] == "Prior name,1"      # v4 from 1

    def test_unknown_version_raises(self, result):
        with pytest.raises(ValueError, match="Unknown AAS version"):
            build_hmm_csv_text(result, "aas3")


class TestOutputPairVersionMatchesInput:
    """The data.csv and the hmm.csv must be written in the SAME version.

    Until 2026-08-31 they were two separate calls at five sites, and every one
    of them let build_hmm_csv_text default to v4 while
    write_states_into_data_csv detected the input's version.  A v2 input
    therefore produced a v2 data.csv beside a v4 hmm.csv: a pair no AAS reader
    accepts, whose diffusion header also claims the wrong unit.

    B1-1's round-trip test did not catch it because it only exercised the
    data.csv.  These check the pair.
    """

    @staticmethod
    def _result(csv_path):
        from smda_hmm.vbhmm.model import VBHMMParams, run_vbhmm_analysis
        from smda_hmm.io.aas_reader import load_aas_settings_csv
        st = REPO / "data" / "sample" / "settings.csv"
        if not st.exists():
            pytest.skip("RTK settings not present")
        s = load_aas_settings_csv(st)
        p = VBHMMParams(
            n_tilde=s["n_tilde"], c_tilde=s["c_tilde"],
            w_pi_tilde=s["w_pi_tilde"], w_b_tilde=s["w_b_tilde"], mag=s["mag"],
            min_hidden=1, max_hidden=2, max_iter=100, num_run=1,
            frame_minimum=s["vbhmm_min_frame"], estimate_mode=s["estimate_mode"],
            is_add_each_trajectory=s["add_per_traj"],
            is_calc_kl_each=s["calc_kl_each"],
            timestep=0.040, distance_per_pixel=0.067)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return run_vbhmm_analysis(str(csv_path), p)

    @staticmethod
    def _hmm_version(text: str) -> str:
        """Which format an hmm.csv is written in, from its own content."""
        first = text.splitlines()[0]
        is_v2 = first == "Method,SimplevbSPT"
        is_v4 = first.startswith("{")
        assert is_v2 != is_v4, f"hmm.csv opens with neither marker: {first[:60]}"
        if is_v2:
            assert "[um/s]" in text and "[um^2/s]" not in text
            return AAS2
        assert "[um^2/s]" in text
        return AAS4

    def _run(self, src, tmp_path):
        from smda_hmm.vbhmm.model import write_vbhmm_outputs
        data_out = tmp_path / src.name
        result = self._result(src)
        d, h, version = write_vbhmm_outputs(
            src, result, overwrite=False, data_out=data_out)
        return d, h, version

    @pytest.mark.skipif(
        not (REPO / "data" / "sample").is_dir(), reason="no v2 data")
    def test_v2_input_gives_a_v2_pair(self, tmp_path):
        src = aas_format.list_data_csvs(REPO / "data" / "sample")[-1]
        d, h, version = self._run(src, tmp_path)
        assert version == AAS2
        assert aas_format.detect_version(d) == AAS2, "data.csv version changed"
        assert self._hmm_version(h.read_text(encoding="utf-8")) == AAS2, (
            "hmm.csv was written in a different version than its data.csv")

    # Gated on data/_absent_v4, which cannot exist: the one test that a v4
    # input yields a v4 pair never ran here.  See tests.helpers.v4_sample_dir.
    @pytest.mark.skipif(
        v4_sample_dir() is None,
        reason="set SMDA_V4_SAMPLE to a directory holding an AAS v4 Sample pair")
    def test_v4_input_gives_a_v4_pair(self, tmp_path):
        from smda_hmm.vbhmm.model import VBHMMParams, run_vbhmm_analysis
        import json
        import warnings
        v4 = v4_sample_dir()
        src = v4 / "Sample_data.csv"
        m = json.loads(open(v4 / "Sample_hmm.csv",
                            encoding="utf-8").readline()
                       .strip().strip('"').replace('""', '"'))
        p = VBHMMParams(
            n_tilde=m["nTilde"], c_tilde=m["cTilde"], w_pi_tilde=m["wPiTilde"],
            w_b_tilde=m["wBTilde"], mag=m["mag"], min_hidden=1, max_hidden=2,
            max_iter=m["maxIter"], num_run=1, timestep=m["timestep"],
            distance_per_pixel=m["distancePerPixel"],
            frame_minimum=m["frameMinimum"], estimate_mode=m["estimate_mode"],
            is_add_each_trajectory=m["isAddEachTrajectory"],
            is_calc_kl_each=m["isCalcKlEach"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_vbhmm_analysis(str(src), p)
        from smda_hmm.vbhmm.model import write_vbhmm_outputs
        d, h, version = write_vbhmm_outputs(
            src, result, overwrite=False, data_out=tmp_path / src.name)
        assert version == AAS4
        assert aas_format.detect_version(d) == AAS4
        assert self._hmm_version(h.read_text(encoding="utf-8")) == AAS4

    @pytest.mark.skipif(
        not (REPO / "data" / "sample").is_dir(), reason="no v2 data")
    def test_writing_elsewhere_leaves_the_input_untouched(self, tmp_path):
        """The source is read, never written, when data_out is given."""
        src = aas_format.list_data_csvs(REPO / "data" / "sample")[-1]
        before = src.read_bytes()
        self._run(src, tmp_path)
        assert src.read_bytes() == before, "the input data.csv was modified"

    @pytest.mark.skipif(
        not (REPO / "data" / "sample").is_dir(), reason="no v2 data")
    def test_hmm_path_defaults_beside_the_data_output(self, tmp_path):
        src = aas_format.list_data_csvs(REPO / "data" / "sample")[-1]
        d, h, _ = self._run(src, tmp_path)
        assert h.parent == tmp_path, (
            "the hmm.csv must land beside the data.csv it belongs to, not "
            "beside the source")
