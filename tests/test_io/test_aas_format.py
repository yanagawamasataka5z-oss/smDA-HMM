"""AAS v2 / v4 format handling.

Background
----------
Until 2026-08-31 smda could not read v2 at all.  ``_load_raw_csv`` decided how
many columns were numeric by looking at the FIRST data row; in v2 that row is
fully populated, so all 18 columns were read as numeric and the load then died
on the first trajectory's final row, whose five state cells are empty by
design::

    ValueError: could not convert string '' to float64 at row 21, column 13

The empty cell is the terminal marker: the state on row t describes the step
t -> t+1, so the last row of a trajectory has no step to describe.  v4 writes 0
there, v2 leaves the cell empty.  Filling those cells with 0 would have made
the file load while destroying the distinction between "no step here" and a
measured state, so the loader validates where NaN is allowed instead.

These tests run against the real AAS output in the repository; they skip
rather than fabricate input when it is absent.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from smda_hmm.io import aas_format
from smda_hmm.io.aas_format import AAS2, AAS4
from tests.helpers import v4_sample_dir

REPO = Path(__file__).resolve().parents[2]
IGOR_V2 = Path(os.environ.get("SMDA_IGOR_SAMPLEDATA", ""))


def _v2_files() -> list[Path]:
    out = list(aas_format.list_data_csvs(REPO / "data" / "sample"))
    if IGOR_V2.is_dir():
        out += sorted(IGOR_V2.glob("*_AAS/*_AAS2.csv"))
    return out


def _v4_files() -> list[Path]:
    """v4 pairs to check, from SMDA_V4_SAMPLE; see tests.helpers.v4_sample_dir.

    This looked in ``data/_absent_v4``, a directory that cannot exist, so the
    v4 half of the format handling was never exercised in this repository and
    the suite reported a clean run regardless.
    """
    d = v4_sample_dir()
    return [d / "Sample_data.csv"] if d else []


V2 = _v2_files()
V4 = _v4_files()

V4_REASON = "set SMDA_V4_SAMPLE to a directory holding an AAS v4 Sample pair"


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not V2, reason="no v2 data in this checkout")
@pytest.mark.parametrize("path", V2, ids=lambda p: p.name[:40])
def test_v2_detected(path):
    assert aas_format.detect_version(path) == AAS2


@pytest.mark.skipif(not V4, reason=V4_REASON)
@pytest.mark.parametrize("path", V4, ids=lambda p: p.name[:40])
def test_v4_detected(path):
    assert aas_format.detect_version(path) == AAS4


def test_unknown_header_raises(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot detect AAS version"):
        aas_format.detect_version(p)


def test_ambiguous_header_raises(tmp_path):
    """A header naming both column families is refused rather than guessed."""
    p = tmp_path / "x.csv"
    p.write_text("Model 1,state(diffusion) 1\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="BOTH"):
        aas_format.detect_version(p)


# ---------------------------------------------------------------------------
# Column layout
# ---------------------------------------------------------------------------

def test_state_columns_are_at_the_same_indices():
    """The positional dstate_col = 11 + dstate works for both versions only
    because the state columns coincide.  If that ever stops being true the
    loader's positional access is wrong for one of them."""
    for dstate in range(1, 6):
        assert aas_format.state_col(dstate) == 11 + dstate


def test_v2_has_no_segment_column():
    """v2's index 17 is Label.  Reading it as v4's Segment is the silent
    corruption this module exists to prevent."""
    assert aas_format.segment_col(AAS2) is None
    assert aas_format.segment_col(AAS4) == 17


@pytest.mark.skipif(not V2, reason="no v2 data")
def test_declared_column_counts_match_the_files():
    for p in V2:
        n = len(open(p, encoding="utf-8").readline().strip().split(","))
        assert n == aas_format.column_count(AAS2) == 18
    for p in V4:
        n = len(open(p, encoding="utf-8").readline().strip().split(","))
        assert n == aas_format.column_count(AAS4) == 19


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------





def _write_v2(path: Path, rows: list[list[str]]) -> None:
    header = (["No", "Time [frame]", "xg [px]", "yg [px]", "sigma x [px]",
               "sigma y [px]", "Imax [a.u.]", "Iback [a.u.]", "a [a.u./px]",
               "b[a.u./px]", "Raw Intensity[au]", "Average Intensity[au/px^2]",
               "Model 1", "Model 2", "Model 3", "Model 4", "Model 5", "Label"])
    lines = [",".join(header)] + [",".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _v2_row(no: int, frame: int, states: list[str]) -> list[str]:
    return ([str(no), str(frame)] + ["1.0"] * 10 + states + ["0"])




# ---------------------------------------------------------------------------
# File pairing
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not V2, reason="no v2 data")
def test_v2_pairs_resolve_without_the_data_suffix():
    """v2 is <stem>.csv <-> <stem>_hmm.csv; the *_data.csv glob never saw it."""
    for p in V2:
        hmm = aas_format.hmm_path_for(p)
        assert hmm is not None and hmm.is_file(), p
        data, hmm2, version = aas_format.resolve_pair(hmm)
        assert data == p and hmm2 == hmm and version == AAS2


@pytest.mark.skipif(not V4, reason=V4_REASON)
def test_v4_pairs_resolve():
    for p in V4:
        hmm = aas_format.hmm_path_for(p)
        if hmm is None:
            continue
        data, _hmm, version = aas_format.resolve_pair(hmm)
        assert data == p and version == AAS4


def test_missing_partner_returns_none_rather_than_a_made_up_path(tmp_path):
    p = tmp_path / "cell.csv"
    p.write_text("Model 1\n1\n", encoding="utf-8")
    assert aas_format.hmm_path_for(p) is None


def test_ambiguous_pair_raises(tmp_path):
    (tmp_path / "c_hmm.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "c.csv").write_text("Model 1\n1\n", encoding="utf-8")
    (tmp_path / "c_data.csv").write_text("state(diffusion) 1\n1\n",
                                         encoding="utf-8")
    with pytest.raises(ValueError, match="Both"):
        aas_format.data_path_for(tmp_path / "c_hmm.csv")


@pytest.mark.skipif(not V2, reason="no v2 data")
def test_listing_finds_v2_and_skips_derived_files():
    folder = V2[0].parent
    found = aas_format.list_data_csvs(folder)
    assert V2[0] in found
    assert not any(f.name.endswith("_hmm.csv") for f in found)


# ---------------------------------------------------------------------------
# Byte-exact round-trip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (V2 + V4), reason="no AAS data")
@pytest.mark.parametrize("path", V2 + V4, ids=lambda p: p.name[:40])
def test_text_round_trip_is_byte_exact(path):
    """Reading and writing back without changing anything must reproduce the
    file exactly, including the empty terminal cells.

    Parsing to float and re-formatting would rewrite recorded measurements the
    analysis never touched, and would turn the v2 terminal marker into
    something else.
    """
    df, _version = aas_format.read_data_csv_text(path)
    out = df.to_csv(index=False, lineterminator="\n")
    assert out == path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Resolving state columns by name
#
# The viewer must open either format without being told which it is: the
# column name already says so.  Before 2026-08-31 four sites in the viewer
# matched the literal "state(diffusion)", so a v2 file loaded fine and then
# reported "No HMM state columns in CSV" because its columns are "Model N".
# ---------------------------------------------------------------------------

class TestStateColumnNames:

    @pytest.mark.parametrize("name,expected", [
        ("state(diffusion) 1", 1),
        ("state(diffusion) 5", 5),
        ("Model 1", 1),
        ("Model 3", 3),
        ("  Model 2  ", 2),
        ("STATE(DIFFUSION) 4", 4),
    ])
    def test_recognised(self, name, expected):
        assert aas_format.parse_state_column(name) == expected

    @pytest.mark.parametrize("name", [
        "No", "Label", "Model", "Imax 1", "Mean Squared Error 1",
        "Average Intensity 1", "Contours [json]", "sigma x 1 [px]",
    ])
    def test_not_a_state_column(self, name):
        """Columns that merely end in a digit must not be mistaken for state
        columns; v4's 'Mean Squared Error 1' sits right next to them."""
        assert aas_format.parse_state_column(name) is None

    @pytest.mark.skipif(not V2, reason="no v2 data")
    def test_v2_columns_resolve(self):
        import pandas as pd
        cols = pd.read_csv(V2[0], nrows=0).columns
        assert aas_format.available_state_counts(cols) == [1, 2, 3, 4, 5]
        assert aas_format.find_state_column(cols, 3) == "Model 3"

    @pytest.mark.skipif(not V4, reason=V4_REASON)
    def test_v4_columns_resolve(self):
        import pandas as pd
        cols = pd.read_csv(V4[0], nrows=0).columns
        assert aas_format.available_state_counts(cols) == [1, 2, 3, 4, 5]
        assert aas_format.find_state_column(cols, 3) == "state(diffusion) 3"

    def test_missing_model_size_returns_none(self):
        assert aas_format.find_state_column(["Model 1", "Model 2"], 4) is None


