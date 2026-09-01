"""smDA-HMM must never write to the file it was given.

This replaces smDA-Python's tests/test_no_viewer_writes.py.  There the concern
was that the viewer might write; here there is no viewer, and the guarantee is
narrower and more central: **re-analysis reads its input and writes elsewhere**.
That is what makes it safe to point at AAS output.

The guarantee is structural rather than a flag that has to stay off.  The
only writers are `write_data_csv` and `write_hmm_csv` in
`smda_hmm.vbhmm.model`, and the only path that reaches them is
`write_vbhmm_outputs`, which the app calls with an explicit `data_out`.  These
tests pin that shape as well as the behaviour, because a future caller could
easily reintroduce an in-place write and every file would still look fine
until someone noticed their AAS output had been replaced.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "smda_hmm"
DATA = REPO / "data" / "sample"

WRITERS = {"write_data_csv", "write_hmm_csv", "_write_text"}
# Where a writer may legitimately be defined or called.
ALLOWED = {
    Path("smda_hmm/vbhmm/model.py"),   # defines them; write_vbhmm_outputs calls them
}


def _modules():
    return sorted(p for p in PKG.rglob("*.py"))


def _calls(tree) -> set[str]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = (f.id if isinstance(f, ast.Name)
                    else f.attr if isinstance(f, ast.Attribute) else None)
            if name:
                out.add(name)
    return out


def test_only_the_model_module_calls_the_writers():
    offenders = []
    for path in _modules():
        rel = path.relative_to(REPO)
        if rel in ALLOWED:
            continue
        hits = _calls(ast.parse(path.read_text(encoding="utf-8"))) & WRITERS
        if hits:
            offenders.append((str(rel), sorted(hits)))
    assert not offenders, (
        f"a writer is called outside {sorted(str(a) for a in ALLOWED)}: "
        f"{offenders}. Route it through write_vbhmm_outputs with an explicit "
        f"data_out instead, so the destination is always chosen deliberately.")


def test_the_app_always_passes_a_destination():
    """write_vbhmm_outputs defaults data_out to the source; the app must not
    take that default, or it would write in place."""
    src = (PKG / "app" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "write_vbhmm_outputs"]
    assert calls, "the app no longer calls write_vbhmm_outputs"
    for call in calls:
        kwargs = {k.arg for k in call.keywords}
        assert "data_out" in kwargs, (
            "write_vbhmm_outputs called without data_out: that writes the "
            "state columns back into the input file")


def test_no_in_place_writer_is_imported():
    """smDA-Python also has write_states_into_data_csv, which writes to the
    path it is given.  It is not part of this package and must not creep in."""
    for path in _modules():
        text = path.read_text(encoding="utf-8")
        assert "write_states_into_data_csv" not in text, (
            f"{path.relative_to(REPO)} references the in-place writer")


@pytest.mark.skipif(not DATA.is_dir(), reason="bundled data not present")
def test_a_real_run_leaves_every_input_byte_identical(tmp_path):
    """The behavioural half.  Structure can be right and a stray open() still
    wrong, so this runs the analysis and compares the bytes."""
    from smda_hmm.io import aas_format
    from smda_hmm.vbhmm.engine import build_params, run
    from smda_hmm.vbhmm.model import write_vbhmm_outputs

    inputs = aas_format.list_data_csvs(DATA)
    assert inputs, "no data.csv in the bundled sample"
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
              for p in DATA.glob("*.csv")}

    params = build_params(
        dict(min_states=1, max_states=3, max_iter=100, n_tilde=1.0,
             c_tilde=0.001, wpi_tilde=1.0, wb_tilde=1.0, mag=10.0,
             add_per_traj=True, calc_kl_each=True), 0.040, 0.067)
    for src in inputs[:2]:
        result = run(str(src), params, seed=42)
        write_vbhmm_outputs(src, result, overwrite=False,
                            data_out=tmp_path / src.name)

    for path, (blob, mtime) in before.items():
        assert path.read_bytes() == blob, f"{path.name} was modified"
        assert path.stat().st_mtime_ns == mtime, f"{path.name} was rewritten"
