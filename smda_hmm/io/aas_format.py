"""AAS data.csv / hmm.csv format: version detection, column layout, file pairing.

Ported from smDA-Python unchanged except that the two movie-to-table
lookups (data_csv_for_movie, data_csv_for_stem) are dropped: they exist
for the viewer, which this package does not include.

This module is the ONE place that knows how the two AAS output formats differ.
Callers ask it which version a file is and where the columns are; they do not
branch on version themselves.

The two formats
---------------
Both put the same twelve localisation columns at indices 0-11 and the five
per-model state columns at indices 12-16.  Only the column NAMES and what
follows index 16 differ:

    index   v4 (19 columns)              v2 (18 columns)
    0-11    No .. Average Intensity      same quantities, different spellings
    12-16   state(diffusion) 1-5         Model 1-5
    17      Mean Squared Error 1         Label
    18      Contours [json]              -

Because the state columns land at the same indices in both, positional access
(``11 + dstate``) resolves correctly for either version.  What does NOT
transfer is index 17: v4 has a numeric per-spot error there, v2 has a label.
Reading v2's Label as v4's Segment is silent corruption, so SEGMENT_COL is
exposed as None for v2 and callers must gate on it.

Terminal marker
---------------
The state on row t describes the step t -> t+1, so a trajectory's last row has
no corresponding step and carries a terminal marker instead: ``0`` in v4, an
EMPTY CELL in v2.  See CLAUDE.md for the full convention.  An empty cell
anywhere else is a malformed file, not a value to be filled in.
"""
from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "AAS2", "AAS4", "STATE_COL_FIRST", "STATE_COL_LAST",
    "state_column_names", "read_data_csv_text", "TERMINAL_MARKER",
    "parse_state_column", "find_state_column",
    "available_state_counts",
    "detect_version", "column_count", "segment_col", "state_col",
    "hmm_path_for", "hmm_output_path_for", "failed_hmm_output_path_for",
    "data_path_for", "resolve_pair",
    "list_data_csvs",
]

AAS2 = "aas2"
AAS4 = "aas4"

# State columns occupy the same indices in both versions (verified below).
STATE_COL_FIRST = 12
STATE_COL_LAST = 16

_N_COLS = {AAS2: 18, AAS4: 19}
# Index 17 is Segment in v4 and Label in v2, so v2 has no segment column.
_SEGMENT_COL = {AAS2: None, AAS4: 17}


def detect_version(csv_path: str | Path) -> str:
    """Return ``AAS2`` or ``AAS4`` for a data.csv, from its header row.

    Basis for the condition
    -----------------------
    The two families are told apart by the name of the state columns:
    v2 spells them ``Model 1``..``Model 5``, v4 spells them
    ``state(diffusion) 1``..``state(diffusion) 5``.  No other header token is
    used, because the remaining names differ only in spacing and unit
    spelling between AAS builds.

    This was checked against every AAS-produced data.csv available in this
    project as of 2026-08-31:

      * v2, 18 columns, ``Model N``  -- smDA-Igor/SampleData/*_AAS/*_AAS2.csv
        (4 files) and data/260830SampleData (8 files).
      * v4, 19 columns, ``state(diffusion) N`` -- data/Sample/Sample_data.csv
        and data/SMLM/{AT1R,Arrb2}/{L1,L2} (8 files).

    A header carrying neither token raises.  If a future AAS revision renames
    the state columns again, extend this function rather than adding a branch
    at a call site -- the point of this module is that the mapping from header
    to layout exists once.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()

    has_model = "Model 1" in header
    has_state = "state(diffusion)" in header
    if has_model and has_state:
        raise ValueError(
            f"Header of {csv_path} contains BOTH 'Model 1' and "
            f"'state(diffusion)'. The version cannot be determined and "
            f"guessing would pick the wrong column layout.\n  {header[:200]}"
        )
    if has_model:
        return AAS2
    if has_state:
        return AAS4
    raise ValueError(
        f"Cannot detect AAS version of {csv_path}: the header has neither "
        f"'Model 1' (v2) nor 'state(diffusion)' (v4).\n  {header[:200]}"
    )


def column_count(version: str) -> int:
    """Number of columns the format defines."""
    return _N_COLS[_check(version)]


def segment_col(version: str) -> int | None:
    """Index of the Segment column, or None when the format has none.

    v2's index 17 is ``Label``, not a segment index; returning None keeps a
    caller from reading it as one.
    """
    return _SEGMENT_COL[_check(version)]


def state_col(dstate: int) -> int:
    """Index of the ``dstate``-state model's state column (1-based dstate).

    Identical for both versions; see the module docstring.
    """
    if not 1 <= dstate <= 5:
        raise ValueError(f"dstate must be 1..5, got {dstate}")
    return STATE_COL_FIRST + dstate - 1


def _check(version: str) -> str:
    if version not in (AAS2, AAS4):
        raise ValueError(f"Unknown AAS version {version!r}; expected "
                         f"{AAS2!r} or {AAS4!r}")
    return version


# ---------------------------------------------------------------------------
# File naming
#
#   v4:  <stem>_data.csv  <->  <stem>_hmm.csv
#   v2:  <stem>.csv       <->  <stem>_hmm.csv
#
# The functions below never invent a path: they return None when the partner
# file does not exist, so the caller decides what to do about it.
# ---------------------------------------------------------------------------

_HMM_SUFFIX = "_hmm.csv"
_FAILED_SUFFIX = "_hmm_FAILED.csv"
_V4_DATA_SUFFIX = "_data.csv"


def hmm_path_for(data_csv: str | Path) -> Path | None:
    """Path of the hmm.csv belonging to *data_csv*, or None if absent."""
    p = Path(data_csv)
    name = p.name
    if name.endswith(_V4_DATA_SUFFIX):
        stem = name[: -len(_V4_DATA_SUFFIX)]
    elif name.endswith(".csv"):
        stem = name[: -len(".csv")]
    else:
        raise ValueError(f"Not a .csv path: {data_csv}")
    cand = p.with_name(stem + _HMM_SUFFIX)
    return cand if cand.is_file() else None


def hmm_output_path_for(data_csv: str | Path) -> Path:
    """Name of the hmm.csv to WRITE for *data_csv*, whether or not it exists.

    Distinct from :func:`hmm_path_for`, which reports an existing partner and
    returns None when there is none.  Call sites used to spell this out inline
    as ``replace("_data.csv", "_hmm.csv")`` followed by a hand-rolled fallback
    for names without the ``_data`` part -- ten copies of the same two lines,
    which is how v2 came to be visible on some paths and not others.
    """
    p = Path(data_csv)
    name = p.name
    if name.endswith(_V4_DATA_SUFFIX):
        stem = name[: -len(_V4_DATA_SUFFIX)]
    elif name.endswith(".csv"):
        stem = name[: -len(".csv")]
    else:
        raise ValueError(f"Not a .csv path: {data_csv}")
    return p.with_name(stem + _HMM_SUFFIX)


def failed_hmm_output_path_for(data_csv: str | Path) -> Path:
    """Name of the ``*_hmm_FAILED.csv`` to write for *data_csv*."""
    p = hmm_output_path_for(data_csv)
    return p.with_name(p.name[: -len(_HMM_SUFFIX)] + _FAILED_SUFFIX)


def data_path_for(hmm_csv: str | Path) -> Path | None:
    """Path of the data.csv belonging to *hmm_csv*, or None if absent.

    Both naming conventions are tried because the hmm.csv name is the same in
    either; the data.csv that exists on disk decides which convention is in
    use.  If both exist the situation is ambiguous and raises rather than
    picking one.
    """
    p = Path(hmm_csv)
    if not p.name.endswith(_HMM_SUFFIX):
        raise ValueError(f"Not an hmm.csv path: {hmm_csv}")
    stem = p.name[: -len(_HMM_SUFFIX)]
    v4 = p.with_name(stem + _V4_DATA_SUFFIX)
    v2 = p.with_name(stem + ".csv")
    found = [c for c in (v4, v2) if c.is_file()]
    if len(found) > 1:
        raise ValueError(
            f"Both {v4.name} and {v2.name} exist next to {p.name}. Which one "
            f"the hmm.csv belongs to cannot be determined; remove or rename "
            f"one of them."
        )
    return found[0] if found else None



def resolve_pair(path: str | Path) -> tuple[Path, Path | None, str]:
    """Resolve any of the pair to ``(data_csv, hmm_csv_or_None, version)``.

    *path* may be either the data.csv or the hmm.csv.  The version is read
    from the data.csv header, never inferred from the file name.
    """
    p = Path(path)
    if p.name.endswith(_HMM_SUFFIX):
        data = data_path_for(p)
        if data is None:
            raise FileNotFoundError(
                f"No data.csv found next to {p}. Looked for "
                f"{p.name[:-len(_HMM_SUFFIX)]}{_V4_DATA_SUFFIX} (v4) and "
                f"{p.name[:-len(_HMM_SUFFIX)]}.csv (v2)."
            )
        return data, p, detect_version(data)
    if not p.is_file():
        raise FileNotFoundError(f"{p} does not exist")
    return p, hmm_path_for(p), detect_version(p)


def list_data_csvs(folder: str | Path, version: str | None = None) -> list[Path]:
    """List the data.csv files in *folder*.

    With *version* given, only that format's naming convention is listed.
    With None, both are listed.  Derived outputs (``*_hmm.csv``,
    ``*_hmm_FAILED.csv``) are excluded, and so is any file whose header does
    not identify it as an AAS data.csv -- a folder may hold unrelated CSVs,
    and silently treating one as a trajectory table would be worse than
    skipping it.
    """
    if version is not None:
        _check(version)
    out: list[Path] = []
    for c in sorted(Path(folder).glob("*.csv")):
        if c.name.endswith(_HMM_SUFFIX) or c.name.endswith(_FAILED_SUFFIX):
            continue
        is_v4_named = c.name.endswith(_V4_DATA_SUFFIX)
        if version == AAS4 and not is_v4_named:
            continue
        if version == AAS2 and is_v4_named:
            continue
        try:
            found = detect_version(c)
        except (ValueError, OSError, UnicodeDecodeError):
            continue
        if version is not None and found != version:
            continue
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Writing back
# ---------------------------------------------------------------------------

# The marker written on a trajectory's final row, where no step follows.
# v4 writes a numeric 0; v2 leaves the cell empty.  They mean the same thing.
TERMINAL_MARKER = {AAS4: "0", AAS2: ""}

_STATE_NAME = {
    AAS4: "state(diffusion) {n}",
    AAS2: "Model {n}",
}


def state_column_names(version: str, n_min: int = 1, n_max: int = 5) -> list[str]:
    """Names of the state columns, spelled the way *version* spells them."""
    tmpl = _STATE_NAME[_check(version)]
    return [tmpl.format(n=n) for n in range(n_min, n_max + 1)]


def read_data_csv_text(path: str | Path):
    """Read a data.csv as TEXT, returning ``(DataFrame_of_str, version)``.

    Every cell is kept as the exact string the file holds, and empty cells stay
    empty rather than becoming NaN.  This is what makes a byte-exact
    round-trip possible: the state columns are the only thing smDA computes,
    so every other column is written back unchanged instead of being parsed to
    float and re-formatted.  Re-formatting would silently alter the recorded
    measurements even when the analysis did not touch them.
    """
    import pandas as pd

    version = detect_version(path)
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
    delimiter = "\t" if "\t" in first else ","
    df = pd.read_csv(path, delimiter=delimiter, dtype=str,
                     keep_default_na=False, na_filter=False)
    return df, version


# ---------------------------------------------------------------------------
# Reading state columns by name
#
# A reader does not need to be told the version: the column name says it.
# v4 spells the K-state model's column "state(diffusion) K", v2 spells it
# "Model K".  Matching both here is what lets the viewer open either format
# without a version switch of its own -- the switch only matters where a file
# is WRITTEN, because then the spelling has to be chosen.
# ---------------------------------------------------------------------------

_STATE_COL_RE = re.compile(r"^\s*(?:state\(diffusion\)|Model)\s+(\d+)\s*$",
                           re.IGNORECASE)


def parse_state_column(name: str) -> int | None:
    """Number of states of the model this column holds, or None.

    ``"state(diffusion) 3"`` and ``"Model 3"`` both give 3.
    """
    m = _STATE_COL_RE.match(str(name))
    return int(m.group(1)) if m else None


def available_state_counts(columns) -> list[int]:
    """Sorted model sizes for which *columns* carries a state column."""
    out = {n for n in (parse_state_column(c) for c in columns) if n}
    return sorted(out)


def find_state_column(columns, n_states: int) -> str | None:
    """The column holding the *n_states*-state assignment, or None.

    Accepts either spelling, so the caller does not branch on version.
    """
    for c in columns:
        if parse_state_column(c) == n_states:
            return c
    return None
