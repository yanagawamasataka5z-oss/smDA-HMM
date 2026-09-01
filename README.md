# smDA-HMM

VB-HMM diffusion-state analysis of single-molecule trajectory tables.

Give it an AAS `data.csv` and the conditions the recording was made under; it
writes an `hmm.csv` and a copy of the `data.csv` with the state columns filled
in, to a folder you choose. **It never writes to the input.**

**What the bundled recordings are** — the receptor, the stimulus, what the
time points correspond to — **is described in the response letter accompanying
the manuscript.** `data/README.md` covers only what the files themselves
contain. See it also for the AAS settings these data were analysed with.

> This README is a stub; the usage sections are being completed alongside the
> packaged build.

## Running from source

```
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

cd smda_scan
cargo build --release
copy target\release\smda_scan.dll python\smda_scan\smda_scan.cp311-win_amd64.pyd
cd ..

start_smda_hmm.bat
```

Then open <http://localhost:8502>. `PYTHONPATH` is not needed: `app.py` puts
the package and the extension on `sys.path` itself.

## Licence

GPL-3.0-or-later. See `LICENSE`.
