"""smda_scan: the Rust VB-HMM engine.

The crate keeps the name it has in smDA-Python, where "scan" meant the
spot-detection pipeline.  None of that is here: this build exposes exactly one
function.  The name is unchanged so the import path matches the upstream
project these results were verified against.
"""
from .smda_scan import run_vbhmm_model_selection

__all__ = ["run_vbhmm_model_selection"]
