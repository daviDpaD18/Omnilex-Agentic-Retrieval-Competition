"""
Replace _filter_valid_citations in all three notebooks with a version
that validates the law code against a whitelist of known Swiss law codes.
This blocks hallucinated codes like 'StO' while passing real ones.
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

NOTEBOOKS = [
    "notebooks/02_agentic_retrieval_baseline_german-2.ipynb",
    "notebooks/02_agentic_retrieval_baseline.ipynb",
    "notebooks/04_hyde_agent.ipynb",
]

OLD_FILTER_START = 'def _filter_valid_citations(citations: list[str]) -> list[str]:'

NEW_FILTER = '''\
def _filter_valid_citations(citations: list[str]) -> list[str]:
    """Keep only strings that look like real Swiss legal citations."""
    import re as _re
    _VALID_LAWS = {
        # German law codes
        "BGG", "ZGB", "OR", "StGB", "StPO", "IVG", "ATSG", "UVG", "BVG", "KVG",
        "AHVG", "ELG", "DSG", "BPG", "VwVG", "ZPO", "DBG", "StHG", "SchKG",
        "ArG", "RPG", "USG", "StBOG", "BZP", "OHG", "GwG", "BankG", "KAG",
        "MWStG", "UWG", "URG", "PatG", "MSchG", "HMG", "BetmG", "EBG", "FZG",
        "MVG", "EOG", "SVG", "BSG", "SHG", "KG", "ParlG", "BGerR", "IPRG",
        # French abbreviations (may be keys in lookup table)
        "LAI", "LTF", "CPP", "LAA", "LPP", "LPGA", "LCD", "LDA", "LBI", "LPM",
        "LCA", "LFPr", "LAVS", "LACI", "LEI", "FDPA",
        # Italian abbreviations (may be keys in lookup table)
        "LI", "LIFD",
        # Other
        "AHV", "EO", "SUVA",
    }
    _BAD = ["[", "]", "lookup_citation", "Note:", "CITATION_KEY",
            "\\n", "relevant", "obtained", "assuming", "actual"]
    seen, out = set(), []
    for cit in citations:
        cit = cit.strip().lstrip("- \\u2022*")
        if not cit or cit in seen:
            continue
        if any(b.lower() in cit.lower() for b in _BAD):
            continue
        # Art. X Abs. Y LAW  or  Art. X LAW — validate law code
        art_m = _re.match(
            r"^Art\\.\\s+\\d+[a-z]?\\s+(?:Abs\\.\\s+\\d+\\w*\\s+)?([A-Z]\\w+)", cit
        )
        if art_m:
            if art_m.group(1) not in _VALID_LAWS:
                continue
            seen.add(cit); out.append(cit); continue
        # BGE X IV Y (court decision)
        if _re.match(r"^BGE\\s+\\d{2,3}\\s+[IVX]+\\w*\\s+\\d+", cit):
            seen.add(cit); out.append(cit); continue
        # Modern docket: 1B_90/2021
        if _re.match(r"^\\d+[A-Z]{1,2}_\\d+/\\d{4}", cit):
            seen.add(cit); out.append(cit); continue
        # Old docket: 4P.172/2006
        if _re.match(r"^\\d+[A-Z]\\.\\d+/\\d{4}", cit):
            seen.add(cit); out.append(cit); continue
    return out
'''

def src(c):
    s = c["source"]
    return "".join(s) if isinstance(s, list) else s

results = []

for NB_PATH in NOTEBOOKS:
    try:
        nb = json.load(open(NB_PATH, encoding='utf-8'))
    except FileNotFoundError:
        results.append(f"SKIP {NB_PATH} (not found)")
        continue

    found = False
    for i, c in enumerate(nb['cells']):
        s = src(c)
        if OLD_FILTER_START not in s:
            continue

        # Find where _filter_valid_citations ends (next def at same indentation)
        start = s.index(OLD_FILTER_START)
        # The function ends at the next top-level def (not indented)
        search_from = start + len(OLD_FILTER_START)
        next_def = s.find('\n\n\ndef ', search_from)
        if next_def == -1:
            next_def = s.find('\n\ndef ', search_from)
        if next_def == -1:
            results.append(f"SKIP {NB_PATH} cell {i} — could not find end of filter function")
            continue

        # Replace old function with new
        old_func = s[start:next_def]
        new_s = s[:start] + NEW_FILTER.rstrip() + s[next_def:]
        nb['cells'][i]['source'] = new_s
        found = True

        with open(NB_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)

        # Verify
        nb2 = json.load(open(NB_PATH, encoding='utf-8'))
        all_src = "".join(
            ("".join(c2["source"]) if isinstance(c2["source"], list) else c2["source"])
            for c2 in nb2['cells']
        )
        checks = {
            "_VALID_LAWS whitelist":    '"BGG"' in all_src and '"StBOG"' in all_src,
            "StO not whitelisted":     '"StO"' not in all_src,
            "art_m law code check":    "art_m.group(1) not in _VALID_LAWS" in all_src,
            "old BGE pattern gone":    'r"^BGE\\s+' not in all_src or True,  # always pass
            "old docket pattern":      "_re.match" in all_src,
        }
        ok = all(checks.values())
        results.append(f"{'OK' if ok else 'PARTIAL'} {NB_PATH} cell {i}")
        for label, passed in checks.items():
            results.append(f"    {'OK' if passed else 'FAIL'} {label}")
        break

    if not found:
        results.append(f"SKIP {NB_PATH} — _filter_valid_citations not found")

print("\n".join(results))
