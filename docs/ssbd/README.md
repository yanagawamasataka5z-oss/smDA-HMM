# SSBD deposit

The imaging data behind this work is deposited in SSBD, the RIKEN repository
for bioimaging data.

| | |
|---|---|
| Accession | `ssbd-repos-000536` |
| URL | <https://ssbd.riken.jp/repository/536/> |
| DOI | <https://doi.org/10.24631/ssbd.repos.2026.08.536> |

## What is here

`Abe_minimal_metadata_template_ja_MY.xlsx` — the metadata workbook submitted
with the deposit. `Abe_minimal_metadata_template_ja_MY.md` is the same content
rendered as Markdown so it can be read on GitHub without downloading anything;
regenerate it with `scripts/ssbd_xlsx_to_md.py` if the workbook changes.

**This is a snapshot, taken 2026-09-02.** SSBD holds the authoritative record
and may be revised after this copy was made, so where the two differ, the
repository is right and this copy is stale. Follow the DOI above for the
current version.

## What is deposited there, and what is here

SSBD holds the complete dataset: the raw movies and the trajectory tables for
all recordings, **995 GB** in total.

This repository bundles **eight cells** of that dataset — trajectory tables,
the results AAS produced for them, and their image sequences — so that the
diffusion-state analysis can be reproduced without a 995 GB download. See
`data/README.md`.
