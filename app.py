"""Entry point: streamlit run app.py"""
import sys
from pathlib import Path

# The package and the built Rust extension both live beside this file.
ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "smda_scan" / "python"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from smda_hmm.app.main import main  # noqa: E402

main()
