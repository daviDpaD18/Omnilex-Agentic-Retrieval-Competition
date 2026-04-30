"""
Offline Phase: build citation_graph.json

Reads court_considerations.csv, groups every consideration chunk by its base
decision ID (strips the ' E. X.X' subsection suffix), extracts every valid
Swiss statute cited in the text of that decision across ALL its chunks, and
saves the result as a compact JSON mapping:

    {"1B_149/2015": ["Art. 221 Abs. 1 StPO", "Art. 212 Abs. 3 StPO", ...], ...}

Runtime: ~5-10 min on 2.5M rows (pure Python, no GPU).
Output size: ~15-50 MB depending on corpus.
"""
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[1]
COURTS_CSV = REPO_ROOT / "data" / "court_considerations.csv"
OUT_PATH   = REPO_ROOT / "data" / "processed" / "citation_graph.json"

# ── Valid Swiss law abbreviations (same whitelist as _filter_valid_citations) ─
VALID_LAWS = {
    "BV", "Cst", "Cost",
    "BGG", "ZGB", "OR", "StGB", "StPO", "IVG", "ATSG", "UVG", "BVG", "KVG",
    "AHVG", "ELG", "DSG", "BPG", "VwVG", "ZPO", "DBG", "StHG", "SchKG",
    "ArG", "RPG", "USG", "StBOG", "BZP", "OHG", "GwG", "BankG", "KAG",
    "MWStG", "UWG", "URG", "PatG", "MSchG", "HMG", "BetmG", "EBG", "FZG",
    "MVG", "EOG", "SVG", "BSG", "SHG", "KG", "ParlG", "BGerR", "IPRG",
    "LAI", "LTF", "CPP", "LAA", "LPP", "LPGA", "LCD", "LDA", "LBI", "LPM",
    "LCA", "LFPr", "LAVS", "LACI", "LEI", "FDPA", "CO", "CC", "CP",
    "LI", "LIFD", "AHV", "EO", "SUVA",
    "EMRK", "AEUV",
}

# Captures: Art. 221 Abs. 1 lit. b StPO  /  Art. 6 Ziff. 1 EMRK  /  Art. 5 EMRK
ART_PAT = re.compile(
    r"Art\.\s+\d+[a-z]?\s+(?:(?:Abs\.|Ziff\.)\s+\d+\w*(?:\s+lit\.\s+\w+)?\s+)?([A-Z]\w+)"
)


def base_id(citation: str) -> str:
    """Strip ' E. X.X' suffix to get the base decision identifier."""
    if " E. " in citation:
        return citation.split(" E. ")[0].strip()
    return citation.strip()


def extract_statutes(text: str) -> list[str]:
    """Return validated statute citation strings found in text."""
    results = []
    for m in ART_PAT.finditer(text):
        law_code = m.group(1)
        if law_code in VALID_LAWS:
            statute = m.group(0).strip().rstrip(".,;:")
            results.append(statute)
    return results


def main() -> None:
    if not COURTS_CSV.exists():
        print(f"ERROR: {COURTS_CSV} not found.")
        sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building citation graph from {COURTS_CSV}")
    print(f"Output → {OUT_PATH}\n")

    graph: dict[str, set[str]] = defaultdict(set)
    rows = 0
    t0 = time.time()

    with open(COURTS_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            citation = (row.get("citation") or "").strip()
            text     = (row.get("text")     or "")

            if not citation:
                rows += 1
                continue

            dec_id = base_id(citation)
            for statute in extract_statutes(text):
                graph[dec_id].add(statute)

            rows += 1
            if rows % 250_000 == 0:
                elapsed = time.time() - t0
                print(f"  {rows:>9,} rows | {len(graph):>7,} decisions | {elapsed:.0f}s elapsed")

    # Convert sets → sorted lists, drop decisions with no statute links
    graph_out = {k: sorted(v) for k, v in graph.items() if v}

    elapsed = time.time() - t0
    print(f"\nFinished: {rows:,} rows → {len(graph_out):,} decisions with statute links "
          f"({elapsed:.0f}s)")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(graph_out, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"Saved {OUT_PATH.name}: {size_mb:.1f} MB")

    # Sample
    sample = list(graph_out.items())[:5]
    print("\nSample entries:")
    for dec, statutes in sample:
        print(f"  {dec}: {statutes[:4]}")


if __name__ == "__main__":
    main()
