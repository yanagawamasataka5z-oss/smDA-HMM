"""Build the portable Windows bundle: embeddable Python + the app, zipped.

    py -3.11 build\\make_bundle.py

Produces `build_output/smDA-HMM-<version>-win64/` and a zip of it under
`build_output/dist/`.  Nothing is installed on the machine and nothing outside
`build_output/` is written.

The result carries its own interpreter and every dependency, so a reviewer
extracts it and runs `smDA-HMM.bat`.

What is deliberately left out
-----------------------------
`data/sample/` also holds the raw image sequences the trajectory tables came
from -- 816 files, 416 MB.  smDA-HMM does not read images, so bundling them
would more than quadruple the download for files the application cannot open.
They stay in the repository, and `data/README.md` says so.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build_output"

PYTHON_VERSION = "3.11.9"
EMBED_URL = (f"https://www.python.org/ftp/python/{PYTHON_VERSION}/"
             f"python-{PYTHON_VERSION}-embed-amd64.zip")
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

PYD = "smda_scan.cp311-win_amd64.pyd"

# Trimmed after install: pip and setuptools are not needed to run, and the
# bundled test suites of the dependencies are dead weight.
PRUNE_DIRS = {"__pycache__", "tests", "test"}
PRUNE_PACKAGES = ("pip", "setuptools", "pkg_resources")

LAUNCHER = """@echo off
rem Start smDA-HMM.  Everything needed is inside this folder; nothing is
rem installed and nothing outside it is touched.
setlocal
cd /d "%~dp0"
echo Starting smDA-HMM.  A browser tab will open at http://localhost:8502
echo Close this window to stop it.
echo.
"%~dp0python\\python.exe" -m streamlit run "%~dp0app\\app.py" --server.port 8502
endlocal
"""

PTH = """python311.zip
.
Lib\\site-packages
..\\app

# Required so pip-installed packages are importable.
import site
"""


def version() -> str:
    ns: dict = {}
    exec((ROOT / "smda_hmm" / "__init__.py").read_text(encoding="utf-8"), ns)
    return ns["__version__"]


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"  downloading {url.split('/')[-1]}")
        urllib.request.urlretrieve(url, dest)
    return dest


def build_python(stage: Path) -> Path:
    py = stage / "python"
    if py.exists():
        shutil.rmtree(py)
    py.mkdir(parents=True)
    archive = fetch(EMBED_URL, OUT / "python-embed.zip")
    with zipfile.ZipFile(archive) as z:
        z.extractall(py)
    (py / f"python{PYTHON_VERSION.replace('.', '')[:3]}._pth").write_text(
        PTH, encoding="utf-8")

    exe = py / "python.exe"
    get_pip = fetch(GET_PIP_URL, OUT / "get-pip.py")
    print("  installing pip")
    subprocess.run([str(exe), str(get_pip), "-q", "--no-warn-script-location"],
                   check=True, cwd=py)
    print("  installing dependencies")
    subprocess.run([str(exe), "-m", "pip", "install", "-q",
                    "--no-warn-script-location", "-r",
                    str(ROOT / "requirements.txt")], check=True, cwd=py)

    print("  pruning")
    subprocess.run([str(exe), "-m", "pip", "uninstall", "-q", "-y",
                    *PRUNE_PACKAGES], cwd=py)
    site = py / "Lib" / "site-packages"
    for name in PRUNE_PACKAGES:
        shutil.rmtree(site / name, ignore_errors=True)
    for d in sorted(site.rglob("*"), key=lambda p: -len(p.parts)):
        if d.is_dir() and d.name in PRUNE_DIRS:
            shutil.rmtree(d, ignore_errors=True)
    return py


def build_app(stage: Path) -> Path:
    app = stage / "app"
    if app.exists():
        shutil.rmtree(app)
    app.mkdir(parents=True)

    shutil.copytree(ROOT / "smda_hmm", app / "smda_hmm",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "app.py", app / "app.py")
    for name in ("LICENSE", "README.md"):
        shutil.copy2(ROOT / name, app / name)

    pyd = ROOT / "smda_scan" / "python" / "smda_scan" / PYD
    if not pyd.exists():
        raise SystemExit(
            f"The Rust extension is not built:\n  {pyd}\n"
            f"Build it with:\n"
            f"  cd smda_scan && cargo build --release\n"
            f"  copy target\\release\\smda_scan.dll python\\smda_scan\\{PYD}")
    dst = app / "smda_scan" / "python" / "smda_scan"
    dst.mkdir(parents=True)
    shutil.copy2(pyd, dst / PYD)
    shutil.copy2(pyd.parent / "__init__.py", dst / "__init__.py")

    # Data, minus the image sequences -- see this module's docstring.
    src_data = ROOT / "data"
    (app / "data" / "sample").mkdir(parents=True)
    shutil.copy2(src_data / "README.md", app / "data" / "README.md")
    kept = 0
    for f in sorted((src_data / "sample").iterdir()):
        if f.is_file():
            shutil.copy2(f, app / "data" / "sample" / f.name)
            kept += 1
    skipped = sum(1 for p in (src_data / "sample").rglob("*")
                  if p.is_file() and p.parent != src_data / "sample")
    print(f"  data: {kept} file(s); skipped {skipped} image-sequence frame(s)")
    return app


def make_zip(stage: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{stage.name}.zip"
    if out.exists():
        out.unlink()
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, p.relative_to(stage.parent))
                n += 1
    raw = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file())
    print(f"\n  files        : {n}")
    print(f"  uncompressed : {raw / 1048576:.0f} MB")
    print(f"  zip          : {out.stat().st_size / 1048576:.0f} MB")
    print(f"  sha256       : {hashlib.sha256(out.read_bytes()).hexdigest()}")
    print(f"  path         : {out}")
    return out


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("This builds a Windows bundle.")
    stage = OUT / f"smDA-HMM-{version()}-win64"
    OUT.mkdir(exist_ok=True)
    print(f"Building {stage.name}")
    build_python(stage)
    build_app(stage)
    (stage / "smDA-HMM.bat").write_text(LAUNCHER, encoding="utf-8")
    make_zip(stage, OUT / "dist")


if __name__ == "__main__":
    main()
