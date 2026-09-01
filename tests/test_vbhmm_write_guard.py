"""Regression tests for the VBHMM write guard (B0, approved 2026-08-30).

The pre-B0 API wrote to a path derived from its input, so pointing an
analysis at reference data destroyed it silently.  These tests pin the
replacement behaviour:

  * builders are pure (no file is created, input is not mutated)
  * writers refuse to clobber an existing file unless overwrite=True
"""

import numpy as np
import pandas as pd
import pytest

from smda_hmm.vbhmm.model import (
    VBHMMResult,
    apply_state_columns,
    build_failed_hmm_csv_text,
    build_vbhmm_failure_log_text,
    write_data_csv,
    write_hmm_csv,
)
from tests.helpers import make_test_vbhmm_params


def _result(min_hidden=1, max_hidden=3, n_rows=4):
    p = make_test_vbhmm_params(min_hidden=min_hidden, max_hidden=max_hidden)
    return VBHMMResult(
        models=[], best_model=max_hidden, params=p,
        state_assignments={k: np.arange(n_rows) % (k + 1)
                           for k in range(min_hidden, max_hidden + 1)},
    )


def _df(n_rows=4):
    return pd.DataFrame({
        "No": [1] * n_rows,
        "Time [frame]": list(range(1, n_rows + 1)),
        "xg [px]": np.arange(n_rows, dtype=float),
        "yg [px]": np.arange(n_rows, dtype=float),
    })


# ---------------------------------------------------------------------------
# Builders are pure
# ---------------------------------------------------------------------------

class TestBuildersArePure:

    def test_apply_state_columns_does_not_mutate_input(self):
        df = _df()
        before = df.copy()
        out = apply_state_columns(df, _result())
        pd.testing.assert_frame_equal(df, before)
        assert out is not df

    def test_apply_state_columns_adds_expected_columns(self):
        out = apply_state_columns(_df(), _result(1, 3))
        for k in (1, 2, 3):
            assert f"state(diffusion) {k}" in out.columns

    def test_builders_create_no_files(self, tmp_path):
        params = make_test_vbhmm_params(min_hidden=1, max_hidden=2)
        build_failed_hmm_csv_text(params, "err", "tb")
        build_vbhmm_failure_log_text("cell", ValueError("x"), "tb", params)
        apply_state_columns(_df(), _result())
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Writers refuse to clobber
# ---------------------------------------------------------------------------

class TestWriteDataCsvGuard:

    def test_writes_when_target_is_new(self, tmp_path):
        out = tmp_path / "x_data.csv"
        write_data_csv(_df(), out)
        assert out.exists()

    def test_refuses_existing_by_default(self, tmp_path):
        out = tmp_path / "x_data.csv"
        out.write_text("REFERENCE DATA\n")
        with pytest.raises(FileExistsError):
            write_data_csv(_df(), out)
        assert out.read_text() == "REFERENCE DATA\n", "input was modified"

    def test_overwrites_when_explicitly_allowed(self, tmp_path):
        out = tmp_path / "x_data.csv"
        out.write_text("OLD\n")
        write_data_csv(_df(), out, overwrite=True)
        assert "Time [frame]" in out.read_text()

    def test_overwrite_is_keyword_only(self, tmp_path):
        with pytest.raises(TypeError):
            write_data_csv(_df(), tmp_path / "a.csv", True)


class TestWriteHmmCsvGuard:

    def test_writes_when_target_is_new(self, tmp_path):
        out = tmp_path / "x_hmm.csv"
        write_hmm_csv("hello\n", out)
        assert out.read_text() == "hello\n"

    def test_refuses_existing_by_default(self, tmp_path):
        out = tmp_path / "x_hmm.csv"
        out.write_text("AAS REFERENCE\n")
        with pytest.raises(FileExistsError):
            write_hmm_csv("new\n", out)
        assert out.read_text() == "AAS REFERENCE\n", "reference was modified"

    def test_overwrites_when_explicitly_allowed(self, tmp_path):
        out = tmp_path / "x_hmm.csv"
        out.write_text("OLD\n")
        write_hmm_csv("NEW\n", out, overwrite=True)
        assert out.read_text() == "NEW\n"

    def test_overwrite_is_keyword_only(self, tmp_path):
        with pytest.raises(TypeError):
            write_hmm_csv("x", tmp_path / "a.csv", True)

    def test_error_message_warns_about_reference_data(self, tmp_path):
        out = tmp_path / "x_hmm.csv"
        out.write_text("AAS\n")
        with pytest.raises(FileExistsError, match="reference data"):
            write_hmm_csv("new\n", out)


# ---------------------------------------------------------------------------
# The scenario that motivated the change
# ---------------------------------------------------------------------------

def test_reference_pair_survives_an_accidental_run(tmp_path):
    """A VBHMM run aimed at reference data must not destroy either file."""
    data = tmp_path / "cell_data.csv"
    hmm = tmp_path / "cell_hmm.csv"
    data.write_text("REFERENCE data.csv\n")
    hmm.write_text("REFERENCE hmm.csv\n")

    with pytest.raises(FileExistsError):
        write_data_csv(_df(), data)
    with pytest.raises(FileExistsError):
        write_hmm_csv("generated\n", hmm)

    assert data.read_text() == "REFERENCE data.csv\n"
    assert hmm.read_text() == "REFERENCE hmm.csv\n"
