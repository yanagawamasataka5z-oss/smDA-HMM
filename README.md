# smDA-HMM

VB-HMM diffusion-state analysis of single-molecule trajectory tables.

Give it an AAS `data.csv` and the conditions the recording was made under; it
writes an `hmm.csv` and a copy of the `data.csv` with the state columns filled
in, to a folder you choose. **It never writes to the input.**

> This README is a stub. The full text — the agreement with AAS expressed
> against the posterior uncertainty of D, the parameter sensitivity table, the
> known unexplained differences, and the bundled-data description — is being
> written.

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
