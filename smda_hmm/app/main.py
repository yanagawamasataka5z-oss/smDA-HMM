"""smDA-HMM: re-analyse an AAS trajectory table with VB-HMM.

One screen.  You give it a data.csv and the conditions the recording was made
under; it writes an hmm.csv and a copy of the data.csv with the state columns
filled in, to a folder you choose.  It never writes to the input.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from smda_hmm import __version__
from smda_hmm.io import aas_format
from smda_hmm.vbhmm.engine import build_params, run as run_vbhmm
from smda_hmm.vbhmm.model import write_vbhmm_outputs

# ---------------------------------------------------------------------------
# Preset
#
# These are the values the bundled data was analysed with: EGFR + EGF,
# well B3, recorded 2022-11-02, AAS v2 (see data/README.md and
# data/RTK/settings.csv).  They are a starting point, not a default that is
# right for any data -- another experiment needs its own settings.csv.
#
# dt and um/px are preset to the same dataset's values.  They describe the
# recording rather than the analysis -- D = c/(4*dt*(a-1)) is linear in dt and
# quadratic in the pixel size through dx^2 -- so the provenance is stated on
# both fields and repeated in the pre-run summary.  Clearing a field stops the
# run rather than falling back to anything.
# ---------------------------------------------------------------------------
PRESET_SOURCE = "bundled RTK data (EGFR + EGF, well B3, 2022-11-02)"
# Measurement conditions of that recording.
PRESET_DT = 0.040
PRESET_UM_PX = 0.067
PRESET = {
    "min_states": 1, "max_states": 5, "max_iter": 100,
    "n_tilde": 1.0, "c_tilde": 0.001, "wpi_tilde": 1.0,
    "wb_tilde": 1.0, "mag": 10.0,
    "add_per_traj": True, "calc_kl_each": True,
}


LABELS = {
    "min_states": "Min States", "max_states": "Max States",
    "max_iter": "Max Iterations", "n_tilde": "n_tilde",
    "c_tilde": "c_tilde", "wpi_tilde": "wPi_tilde", "wb_tilde": "wB_tilde",
    "mag": "mag", "add_per_traj": "Add prior per trajectory",
    "calc_kl_each": "Calc KL per trajectory",
}

DEFAULTS = {
    "in_kind": "Folder", "in_file": "", "in_folder": "", "out_folder": "",
    "dt": PRESET_DT, "um_px": PRESET_UM_PX, **PRESET,
}


def _init_state() -> None:
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    # A Browse button cannot write a key whose widget already exists this run,
    # so the pickers stage their result and it is applied here, first.
    for key in ("in_file", "in_folder", "out_folder"):
        pick = "_pick_" + key
        if st.session_state.get(pick):
            st.session_state[key] = st.session_state.pop(pick)


def _pick_folder(title: str) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title=title)
        root.destroy()
        return path or ""
    except Exception:      # noqa: BLE001 - no display, or tk absent
        return ""


def _pick_file(title: str) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title=title, filetypes=[("CSV", "*.csv")])
        root.destroy()
        return path or ""
    except Exception:      # noqa: BLE001
        return ""


def check_destination(inputs, out_dir) -> list[str]:
    """Problems that would let the run write over an input.  Empty is fine.

    The folder test is the common accident; the per-file test is the actual
    invariant, and resolve() settles symlinks and "." components too.  A
    subfolder of the input folder is allowed: listing is not recursive, so
    results written there are not picked up as inputs by a later run.
    """
    problems = []
    out = Path(out_dir).resolve()
    if out in {Path(p).resolve().parent for p in inputs}:
        problems.append(
            f"The output folder is the input folder. Results would replace "
            f"the files being re-analysed. A subfolder such as "
            f"`{out / 'reanalysis'}` is fine.")
    for p in inputs:
        src = Path(p).resolve()
        if (out / src.name).resolve() == src:
            problems.append(f"`{src.name}` would be written over itself.")
    return problems


def _inputs() -> list[str]:
    st.subheader("1. Input")
    kind = st.radio("Source", ["Folder", "Single file"], horizontal=True,
                    key="in_kind")
    if kind == "Single file":
        c1, c2 = st.columns([4, 1])
        c1.text_input("data.csv", key="in_file",
                      placeholder="Browse, or paste a path")
        with c2:
            st.write("")
            if st.button("Browse", key="b_file"):
                sel = _pick_file("Select data.csv")
                if sel:
                    st.session_state["_pick_in_file"] = sel
                    st.rerun()
        p = st.session_state["in_file"]
        found = [p] if p and os.path.isfile(p) else []
    else:
        c1, c2 = st.columns([4, 1])
        c1.text_input("Folder", key="in_folder",
                      placeholder="Browse, or paste a path")
        with c2:
            st.write("")
            if st.button("Browse", key="b_in"):
                sel = _pick_folder("Select the folder holding the data.csv")
                if sel:
                    st.session_state["_pick_in_folder"] = sel
                    st.rerun()
        d = st.session_state["in_folder"]
        found = ([str(x) for x in aas_format.list_data_csvs(d)]
                 if d and os.path.isdir(d) else [])

    if found:
        versions = {}
        for f in found:
            try:
                versions[aas_format.detect_version(f)] = \
                    versions.get(aas_format.detect_version(f), 0) + 1
            except (ValueError, OSError):
                versions["unreadable"] = versions.get("unreadable", 0) + 1
        st.success(
            f"{len(found)} file(s): "
            + ", ".join(f"{n} x {v}" for v, n in sorted(versions.items()))
            + ". Each output is written in the same format as its input.")
    elif st.session_state["in_file"] or st.session_state["in_folder"]:
        st.warning("No AAS data.csv found there.")
    return found


def _output(found) -> str:
    st.subheader("2. Output")
    c1, c2 = st.columns([4, 1])
    c1.text_input("Folder", key="out_folder",
                  placeholder="Must not be the input folder")
    with c2:
        st.write("")
        if st.button("Browse", key="b_out"):
            sel = _pick_folder("Select the output folder")
            if sel:
                st.session_state["_pick_out_folder"] = sel
                st.rerun()
    out = st.session_state["out_folder"]
    st.caption("The input files are read, never written.")
    if found and out:
        for msg in check_destination(found, out):
            st.error(msg)
    return out


def _parameters() -> tuple[float | None, float | None, dict]:
    st.subheader("3. Measurement conditions")
    st.caption(
        f"Preset to the {PRESET_SOURCE}: **{PRESET_DT:g} s** and "
        f"**{PRESET_UM_PX:g} um/px**. These describe the recording rather "
        f"than the analysis; every diffusion coefficient scales with them.")
    _scale_help = (f"Preset from the {PRESET_SOURCE}. Not stored in the "
                   f"trajectory CSV and not derivable from it.")
    c1, c2 = st.columns(2)
    dt = c1.number_input("dt [s]", min_value=0.0001, max_value=10.0,
                         format="%.4f", key="dt", help=_scale_help)
    um = c2.number_input("um/px", min_value=0.0001, max_value=10.0,
                         format="%.4f", key="um_px", help=_scale_help)

    st.subheader("4. VB-HMM parameters")
    st.caption(f"Preset to the values used for the {PRESET_SOURCE}. "
               f"Another experiment needs its own; check its settings.csv.")
    c1, c2, c3 = st.columns(3)
    p = {
        "min_states": c1.number_input("Min States", 1, 10, key="min_states"),
        "max_states": c2.number_input("Max States", 1, 10, key="max_states"),
        "max_iter": c3.number_input("Max Iterations", 10, 1000,
                                    key="max_iter"),
    }
    with st.expander(f"Hyperparameters — preset from {PRESET_SOURCE}"):
        h1, h2 = st.columns(2)
        p["n_tilde"] = h1.number_input("n_tilde", 0.001, 100.0, key="n_tilde")
        p["wpi_tilde"] = h1.number_input("wPi_tilde", 0.001, 100.0,
                                         key="wpi_tilde")
        p["mag"] = h1.number_input("mag", 1.0, 1000.0, key="mag")
        p["c_tilde"] = h2.number_input("c_tilde", 0.0001, 10.0,
                                       format="%.4f", key="c_tilde")
        p["wb_tilde"] = h2.number_input("wB_tilde", 0.0001, 10.0,
                                        format="%.4f", key="wb_tilde")
        p["add_per_traj"] = st.checkbox("Add prior per trajectory",
                                        key="add_per_traj")
        p["calc_kl_each"] = st.checkbox("Calc KL per trajectory",
                                        key="calc_kl_each")
    return dt, um, p


def _summary(found, out, dt, um, p) -> None:
    """Everything the run will use, before it runs.

    Values and their provenance, and nothing else.  The point of this screen is
    to let someone see what is about to run; what each parameter would do if it
    were wrong is not something a reader can act on here, so it is not shown.
    """
    st.subheader("5. Check before running")
    rows = [
        {"Parameter": "Input", "Value": f"{len(found)} file(s)", "Source": ""},
        {"Parameter": "Output folder", "Value": out or "(not set)",
         "Source": ""},
        {"Parameter": "dt [s]",
         "Value": "(cleared)" if dt is None else f"{dt:g}",
         "Source": "preset" if dt == PRESET_DT else "entered"},
        {"Parameter": "um/px",
         "Value": "(cleared)" if um is None else f"{um:g}",
         "Source": "preset" if um == PRESET_UM_PX else "entered"},
    ]
    for key, value in p.items():
        rows.append({
            "Parameter": LABELS[key], "Value": str(value),
            "Source": "preset" if value == PRESET[key] else "entered"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="content")
    st.caption(f"Rows marked *preset* hold the {PRESET_SOURCE} values.")


def _run(found, out, dt, um, p) -> None:
    if dt is None or um is None:
        missing = " and ".join(
            n for n, v in (("dt [s]", dt), ("um/px", um)) if v is None)
        st.error(
            f"**{missing} is empty.** These are conditions of the recording, "
            f"not stored in the CSV and not derivable from it. D is "
            f"`c / (4*dt*(a-1))` and scales directly with them, so a wrong "
            f"value biases every diffusion coefficient without anything "
            f"looking wrong — which is why an empty field stops the run "
            f"instead of falling back. The {PRESET_SOURCE} was "
            f"recorded at {PRESET_DT:g} s and {PRESET_UM_PX:g} um/px. "
            f"Nothing was written.")
        return
    if not found:
        st.error("No input selected.")
        return
    if not out:
        st.error("**Output folder not set.** Results go to a folder you "
                 "choose so the input is left untouched. Nothing was written.")
        return
    problems = check_destination(found, out)
    if problems:
        for m in problems:
            st.error(m)
        return

    os.makedirs(out, exist_ok=True)
    clash = []
    for src in found:
        d = Path(out) / Path(src).name
        clash += [str(x) for x in (d, aas_format.hmm_output_path_for(d))
                  if x.exists()]
    if clash:
        st.error(
            "**The output folder already holds results with these names.**\n\n"
            + "\n".join(f"- `{os.path.basename(c)}`" for c in clash[:20])
            + "\n\nNothing was written and the inputs are untouched. Use an "
              "empty folder, or move these aside.")
        return

    params = build_params(p, float(dt), float(um))
    bar = st.progress(0.0)
    note = st.empty()
    done, failed = [], []
    for i, src in enumerate(found, start=1):
        name = Path(src).name
        note.text(f"{name} ({i}/{len(found)})")
        try:
            result = run_vbhmm(src, params, seed=42)
            data_p, hmm_p, version = write_vbhmm_outputs(
                src, result, overwrite=False,
                data_out=Path(out) / name)
            done.append({"File": name, "Format": version,
                         "Best N": result.best_model,
                         "data.csv": data_p.name, "hmm.csv": hmm_p.name})
        except Exception as exc:                      # noqa: BLE001
            failed.append({"File": name, "Error": f"{type(exc).__name__}: {exc}"})
        bar.progress(i / len(found))
    note.empty()

    if done:
        st.success(f"{len(done)} file(s) written to `{out}`")
        st.dataframe(pd.DataFrame(done), hide_index=True, width="content")
    if failed:
        st.error(f"{len(failed)} file(s) failed")
        st.dataframe(pd.DataFrame(failed), hide_index=True, width="content")


def main() -> None:
    st.set_page_config(page_title="smDA-HMM", layout="wide")
    _init_state()
    st.title("smDA-HMM")
    st.caption(
        f"v{__version__} — VB-HMM diffusion-state analysis of "
        f"single-molecule trajectory tables. Reads an AAS data.csv, writes "
        f"an hmm.csv and a state-filled data.csv. The input is never modified.")

    found = _inputs()
    out = _output(found)
    dt, um, p = _parameters()
    _summary(found, out, dt, um, p)

    st.divider()
    if st.button("Run", type="primary"):
        _run(found, out, dt, um, p)


if __name__ == "__main__":
    main()
