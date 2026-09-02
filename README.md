# smDA-HMM

VB-HMM classification of single-molecule diffusion states.

Give it a single-molecule trajectory table and the conditions the recording was
made under. It fits hidden Markov models with one to five diffusion states,
selects among them, and writes the result as an `hmm.csv` together with a copy
of the trajectory table carrying the per-step state assignments.

**It never writes to the file you give it.** Results go to a folder you choose,
so a table produced by another program can be re-analysed without being
altered.

This is the analysis half of smDA. Spot detection and tracking — which produced
the trajectory tables — are done by AAS and are not part of this package.

---

## What is here, and what is not

| | |
|---|---|
| **In** | Reading AAS trajectory tables (v2 and v4), VB-HMM inference, model selection over 1–5 states, writing `hmm.csv` and the state-filled table |
| **Out** | Spot detection, tracking, image handling, trajectory visualisation, movie overlays |

A reviewer wanting to reproduce the diffusion-state results in the manuscript
needs this package and the trajectory tables. Reproducing the *detection and
tracking* additionally requires AAS, which is not distributable here.

## Getting started

### Packaged build (nothing to install)

1. Download `smDA-HMM-<version>-win64.zip` from the
   [Releases](https://github.com/yanagawamasataka5z-oss/smDA-HMM/releases) page.
2. Extract it anywhere — a USB stick is fine. Nothing is installed, no
   administrator rights are needed, and nothing outside the extracted folder is
   written to.
3. Double-click **`smDA-HMM.bat`**.
4. A browser tab opens at <http://localhost:8502>. Closing the console window
   stops the program.

The bundle carries its own Python interpreter and every dependency. It does
not use, or interfere with, any Python already on the machine.

### From source

Requires Python 3.11 and a Rust toolchain.

```
git clone https://github.com/yanagawamasataka5z-oss/smDA-HMM.git
cd smDA-HMM

py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

cd smda_scan
cargo build --release
copy target\release\smda_scan.dll python\smda_scan\smda_scan.cp311-win_amd64.pyd
cd ..

start_smda_hmm.bat
```

Then open <http://localhost:8502>. The port differs from smDA-Python's 8501 so
that both can run at once.

`PYTHONPATH` does not need to be set: `app.py` puts the package and the
extension on `sys.path` itself. `start_smda_hmm.bat` uses `.venv` and stops
with an explanation if it is missing, rather than falling back to whatever
Python is on `PATH`.

To run the tests: `.venv\Scripts\python -m pytest`.

## Using it

The interface is one screen, top to bottom.

**1. Input.** A folder, or a single file. Both AAS naming conventions are
recognised (`*_data.csv` for v4, `*.csv` for v2), and each file's format is
read from its own header — there is nothing to select. Files that are not AAS
trajectory tables are skipped rather than misread.

**2. Output.** A folder, which must not be the input folder. A subfolder of it
is fine. If a result of the same name is already there, the run stops and says
which files are in the way instead of replacing them.

**3. Measurement conditions.** `dt` in seconds and the camera pixel size in
micrometres. They describe the recording rather than the analysis, and are not
stored in the trajectory table, so they cannot be recovered from it. Both are
filled in with the values for the bundled data. Clearing either stops the run.

**4. VB-HMM parameters.** The number of states to search over, the iteration
limit, and the prior hyperparameters. All are filled in with the values the
bundled data was analysed with.

**5. Check before running.** Every value the run will use, in one table, each
marked as a preset or as something you entered.

Then **Run**. Each input produces two files in the output folder: the
trajectory table with its state columns filled in, and the `hmm.csv`. Both are
written in the same format as the input.

### Output format follows the input

You cannot choose whether results come out as v2 or v4, and this is deliberate.
The formats do not carry the same columns — v4 has a per-spot fitting error and
a contour field where v2 has a label — so converting between them would mean
inventing values for columns the source does not have, and writing them into a
file that reads as measured data. This package computes the state columns;
everything else in the table belongs to the program that produced it and is
copied through untouched.

### Checking a result

`data/sample/` holds eight trajectory tables together with the `hmm.csv` that
AAS produced for each. Running them through smDA-HMM with the values already
filled in reproduces AAS's result, and `data/README.md` describes the files and
lists the settings. **What the recordings are — the receptor, the stimulus,
what the time points correspond to — is described in the response letter
accompanying the manuscript.**

What "reproduces" means here, measured on those eight:

- the selected number of states matches AAS on all eight;
- across the 24 resulting states, the difference in *D* has a median of
  **0.0043** and a maximum of **0.42** standard deviations of *D* itself, and
  no state differs by as much as one standard deviation.

The comparison is against the width of *D*'s own posterior, rather than as a
percentage, because a fast state and a slow state are measured to very
different absolute precision. For a gamma posterior with shape *a*, the
standard deviation is `D / sqrt(a - 2)`; AAS reports it in its own `hmm.csv`,
and the two agree to a median of 0.004 %.

One cell, `egfr-EGF_t000128`, differs more than the rest — about 0.4
standard deviations, against under 0.03 for the others. This is not a
disagreement about the answer: AAS stopped iterating slightly before the fixed
point on that cell. Restarting the iteration from the parameters in AAS's own
`hmm.csv` moves towards the same fixed point smDA-HMM reaches, which was
checked directly. The remaining difference is how far short of it AAS stopped,
and it is still well inside the measurement uncertainty.

## Repository layout

```
smda_hmm/          Python package
  vbhmm/model.py     the model: preprocessing, priors, initialisation,
                     VBEM, the lower bound, state assignment, file writing
  vbhmm/engine.py    running it: parameter assembly, the call into Rust,
                     converting the result back
  io/                reading and writing AAS tables, version detection
  app/               the interface
smda_scan/         Rust extension — the VB-HMM computation itself
data/sample/       eight trajectory tables with AAS's own results
tests/
```

`smda_scan` is where the inference actually runs; the Python in
`vbhmm/model.py` implements the same computation and serves as the readable
reference. The two were checked against each other and agree to 2.4e-08 ppm
with identical iteration counts.

The crate name comes from smDA-Python, where "scan" referred to the
spot-detection pipeline. **No detection or tracking code is present in this
build** — the crate exposes a single function, `run_vbhmm_model_selection`. The
name is unchanged so that the import path matches the upstream project these
results were verified against.

## Method

The inference follows Persson et al., *Nature Methods* **10**, 265–269 (2013)
as implemented in AAS, applied to diffusion states as in Yanagawa et al.,
*Science Signaling* (2018), [doi:10.1126/scisignal.aao1917][yanagawa].

Hiroshima et al., *Journal of Molecular Biology* (2018),
[doi:10.1016/j.jmb.2018.02.018][hiroshima], use the same variational scheme
with a different emission model, for fitting intensities rather than
displacements. That analysis is not part of this package.

[yanagawa]: https://doi.org/10.1126/scisignal.aao1917
[hiroshima]: https://doi.org/10.1016/j.jmb.2018.02.018

Trajectories are reduced to squared displacements; the model has a Dirichlet
prior on the
initial-state and transition distributions and a gamma prior on the diffusion
rate; states are assigned by the per-frame maximum of the posterior. The
diffusion coefficient of state *i* is reported as its posterior expectation,

```
D_i = c_i / [4 (a_i - 1) dt]
```

Initialisation is a deterministic variant of K-means++ — median seeding
followed by farthest-point selection — matching what AAS does, so a given input
and parameter set always give the same answer. Nothing in the analysis draws on
a random number generator.

## Related deposits

| Where | What |
|---|---|
| [SSBD `ssbd-repos-000536`](https://doi.org/10.24631/ssbd.repos.2026.08.536) | The imaging data: raw movies and the trajectory tables for every recording, 995 GB. `docs/ssbd/` holds the submitted metadata. |
| This repository | The VB-HMM implementation, and eight cells of that dataset to run it on. |
| [smDA-Igor](https://github.com/yanagawamasataka5z-oss/smDA-Igor) | The Igor implementation. |

## Licence

GPL-3.0-or-later. See `LICENSE`.

GPL v3 rather than the v2 of the Igor implementation: Streamlit and its
dependency pyarrow are Apache-2.0, which the Free Software Foundation holds
incompatible with GPL-2.0-only, and the packaged build distributes them.

## Citing

Please cite the accompanying manuscript, and for the method:

> Yanagawa et al., *Science Signaling* (2018).
> <https://doi.org/10.1126/scisignal.aao1917>
