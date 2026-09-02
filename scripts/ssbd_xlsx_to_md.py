"""Render the SSBD metadata workbook as Markdown, so GitHub can display it.

    py scripts\\ssbd_xlsx_to_md.py

Reads `docs/ssbd/*.xlsx` and writes a `.md` beside it.  Re-run after the
workbook changes.

The xlsx is the record of what was sent to SSBD; this is a rendering of it.
Values are copied verbatim -- not reformatted, not tidied, not filled in.  A
blank cell stays blank and a typo stays a typo: if the two ever disagree, a
reader cannot tell which to believe, and the xlsx is the one that was sent.

Uses only the standard library.  openpyxl would read this in a few lines, but
this repository ships four dependencies on purpose and a documentation build
is not a reason to add a fifth.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _shared_strings(z: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    out = []
    for si in root.findall("m:si", NS):
        # A cell's text can be split across runs; join them in document order.
        out.append("".join(t.text or "" for t in si.iter(
            "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")))
    return out


def _sheets(z: zipfile.ZipFile) -> list[tuple[str, str]]:
    """(sheet name, part path), in the workbook's own order."""
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    target = {r.get("Id"): r.get("Target")
              for r in rels.findall(f"{{{REL_NS}}}Relationship")}
    out = []
    for s in wb.find("m:sheets", NS):
        rid = s.get(f"{{{NS['r']}}}id")
        part = target[rid]
        out.append((s.get("name"),
                    part if part.startswith("xl/") else "xl/" + part))
    return out


def _col(ref: str) -> int:
    """Zero-based column index from a cell reference such as 'AB12'."""
    n = 0
    for ch in ref:
        if not ch.isalpha():
            break
        n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


def _rows(z: zipfile.ZipFile, part: str, shared: list[str]) -> list[list[str]]:
    root = ET.fromstring(z.read(part))
    data = root.find("m:sheetData", NS)
    rows = []
    for row in data.findall("m:row", NS):
        cells: dict[int, str] = {}
        for c in row.findall("m:c", NS):
            v = c.find("m:v", NS)
            t = c.get("t")
            if t == "s":
                text = shared[int(v.text)] if v is not None else ""
            elif t == "inlineStr":
                is_el = c.find("m:is", NS)
                text = "".join(x.text or "" for x in is_el.iter(
                    "{http://schemas.openxmlformats.org/spreadsheetml/2006/"
                    "main}t")) if is_el is not None else ""
            else:
                text = v.text if v is not None else ""
            if text:
                cells[_col(c.get("r", "A1"))] = text
        rows.append([cells.get(i, "") for i in range(max(cells) + 1)]
                    if cells else [])
    return rows


def _md_table(rows: list[list[str]]) -> list[str]:
    rows = [r for r in rows if any(x.strip() for x in r)]
    if not rows:
        return ["*(empty)*", ""]
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", "<br>").strip()

    head, *body = rows
    out = ["| " + " | ".join(esc(c) for c in head) + " |",
           "|" + "|".join(["---"] * width) + "|"]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in body]
    out.append("")
    return out


def convert(xlsx: Path) -> Path:
    with zipfile.ZipFile(xlsx) as z:
        shared = _shared_strings(z)
        parts = _sheets(z)
        lines = [f"# {xlsx.stem}", "",
                 "Rendered from the workbook of the same name so it can be "
                 "read on GitHub. **The `.xlsx` is the record**; this is a "
                 "copy of its contents, verbatim — blanks and all. Regenerate "
                 "with `scripts/ssbd_xlsx_to_md.py` after editing the "
                 "workbook.", ""]
        for name, part in parts:
            rows = _rows(z, part, shared)
            lines += [f"## {name}", ""] + _md_table(rows)
    out = xlsx.with_suffix(".md")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    found = [p for p in sorted((root / "docs" / "ssbd").glob("*.xlsx"))
             if not p.name.startswith("~$")]   # Excel lock files
    if not found:
        raise SystemExit("No .xlsx in docs/ssbd/")
    for x in found:
        out = convert(x)
        print(f"{x.name} -> {out.name} "
              f"({len(out.read_text(encoding='utf-8').splitlines())} lines)")


if __name__ == "__main__":
    main()
