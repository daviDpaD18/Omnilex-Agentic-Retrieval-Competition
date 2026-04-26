import json

NB_PATH = "c:/Users/david/Desktop/project_ssl/Omnilex-Agentic-Retrieval-Competition/notebooks/02_agentic_retrieval_baseline.ipynb"

SOURCE = [
    "# ── Index Granularity & Citation Mismatch Diagnosis ─────────────────────\n",
    "# DO NOT FIX ANYTHING HERE — report findings only\n",
    "\n",
    "PROBE_QUERY = 'Untersuchungshaft Verhältnismäßigkeit'\n",
    "\n",
    "GOLD_LAWS   = ['Art. 221 Abs. 1 StPO', 'Art. 100 Abs. 1 BGG', 'Art. 37 Abs. 1 StBOG']\n",
    "GOLD_COURTS = ['BGE 137 IV 122 E. 6.2', '1B_90/2021 E. 2.1', '7B_496/2025 E. 3.2']\n",
    "\n",
    "SEP = '=' * 70\n",
    "sep = '-' * 70\n",
    "\n",
    "print(SEP)\n",
    "print('  STEP 1 — RAW INDEX RESULTS')\n",
    "print(SEP)\n",
    "\n",
    "# ── 1a. Raw search results ────────────────────────────────────────────────\n",
    "for label, index in [('LAWS', laws_index), ('COURTS', courts_index)]:\n",
    "    print(f'\\n{sep}')\n",
    "    print(f'  {label} INDEX — top-3 raw results for: \"{PROBE_QUERY}\"')\n",
    "    print(sep)\n",
    "    raw = index.search(PROBE_QUERY, top_k=3, return_scores=True)\n",
    "    if not raw:\n",
    "        print('  (no results returned)')\n",
    "    for i, doc in enumerate(raw):\n",
    "        print(f'\\n  Result {i+1}:')\n",
    "        for k, v in doc.items():\n",
    "            val_str = str(v)\n",
    "            if len(val_str) > 300:\n",
    "                val_str = val_str[:300] + ' ... [truncated]'\n",
    "            print(f'    {k:20s} = {val_str}')\n",
    "\n",
    "# ── 1b. Citation inventory ────────────────────────────────────────────────\n",
    "print(f'\\n{SEP}')\n",
    "print('  STEP 1b — CITATION INVENTORY')\n",
    "print(SEP)\n",
    "\n",
    "for label, index in [('LAWS', laws_index), ('COURTS', courts_index)]:\n",
    "    all_cits = [doc.get('citation', '') for doc in index.documents]\n",
    "    unique   = list(dict.fromkeys(all_cits))  # preserve insertion order, deduplicate\n",
    "    print(f'\\n  {label} index')\n",
    "    print(f'    Total documents   : {len(index.documents):,}')\n",
    "    print(f'    Unique citations  : {len(set(all_cits)):,}')\n",
    "    print(f'    10 example citation strings:')\n",
    "    for c in unique[:10]:\n",
    "        print(f'      · {repr(c)}')\n",
    "\n",
    "# ── STEP 2 — Compare against gold ────────────────────────────────────────\n",
    "print(f'\\n{SEP}')\n",
    "print('  STEP 2 — GOLD CITATION LOOKUP')\n",
    "print(SEP)\n",
    "\n",
    "import difflib\n",
    "\n",
    "def find_closest(target: str, candidates: list, n: int = 3) -> list:\n",
    "    return difflib.get_close_matches(target, candidates, n=n, cutoff=0.0)\n",
    "\n",
    "for label, index, gold_list in [\n",
    "    ('LAWS',   laws_index,   GOLD_LAWS),\n",
    "    ('COURTS', courts_index, GOLD_COURTS),\n",
    "]:\n",
    "    all_cits = [doc.get('citation', '') for doc in index.documents]\n",
    "    cit_set  = set(all_cits)\n",
    "    print(f'\\n  {label} index')\n",
    "    print(sep)\n",
    "    for gold in gold_list:\n",
    "        exact = gold in cit_set\n",
    "        print(f'\\n    Gold : {repr(gold)}')\n",
    "        print(f'    Exact match in index : {\"YES\" if exact else \"NO\"}')\n",
    "        if not exact:\n",
    "            closest = find_closest(gold, all_cits, n=3)\n",
    "            print(f'    Closest matches      :')\n",
    "            for m in closest:\n",
    "                print(f'      · {repr(m)}')\n",
    "            # Also try prefix search\n",
    "            # e.g. 'Art. 221' -> find all citations starting with 'Art. 221'\n",
    "            prefix = gold.split(' E. ')[0].strip()  # strip judgment paragraph\n",
    "            prefix_matches = [c for c in all_cits if c.startswith(prefix)][:5]\n",
    "            if prefix_matches:\n",
    "                print(f'    Prefix ({repr(prefix)}) hits :')\n",
    "                for m in prefix_matches:\n",
    "                    print(f'      · {repr(m)}')\n",
    "\n",
    "# ── STEP 3 — Diagnosis ───────────────────────────────────────────────────\n",
    "print(f'\\n{SEP}')\n",
    "print('  STEP 3 — GRANULARITY DIAGNOSIS')\n",
    "print(SEP)\n",
    "\n",
    "# Sample document lengths to infer granularity\n",
    "for label, index in [('LAWS', laws_index), ('COURTS', courts_index)]:\n",
    "    sample_docs = index.documents[:20]\n",
    "    text_lens   = [len(doc.get('text', '')) for doc in sample_docs]\n",
    "    avg_len     = sum(text_lens) / len(text_lens) if text_lens else 0\n",
    "    min_len     = min(text_lens) if text_lens else 0\n",
    "    max_len     = max(text_lens) if text_lens else 0\n",
    "\n",
    "    # Check citation format patterns\n",
    "    sample_cits = [doc.get('citation', '') for doc in sample_docs]\n",
    "    has_abs     = any('Abs.' in c for c in sample_cits)\n",
    "    has_art     = any('Art.' in c for c in sample_cits)\n",
    "    has_bge     = any('BGE'  in c for c in sample_cits)\n",
    "    has_e_dot   = any(' E. ' in c for c in sample_cits)  # judgment paragraph marker\n",
    "\n",
    "    print(f'\\n  {label} index')\n",
    "    print(f'    Text length (sample 20) — avg: {avg_len:.0f}  min: {min_len}  max: {max_len}')\n",
    "    print(f'    Citation patterns in sample:')\n",
    "    print(f'      contains \"Art.\"  : {has_art}')\n",
    "    print(f'      contains \"Abs.\"  : {has_abs}')\n",
    "    print(f'      contains \"BGE\"   : {has_bge}')\n",
    "    print(f'      contains \" E. \"  : {has_e_dot}  (judgment paragraph granularity)')\n",
    "    print(f'    Sample citation strings:')\n",
    "    for c in sample_cits[:5]:\n",
    "        print(f'      · {repr(c)}')\n",
    "\n",
    "print(f'\\n{SEP}')\n",
    "print('  END OF DIAGNOSIS — no changes made')\n",
    "print(SEP)\n",
]

HEADER_SOURCE = [
    "## 11. Index Granularity & Citation Mismatch Diagnosis\n",
    "\n",
    "Inspects raw index contents and checks whether gold citation strings are "
    "retrievable as-is. Does **not** modify anything."
]

nb = json.load(open(NB_PATH, encoding="utf-8"))

header_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": HEADER_SOURCE
}
diag_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": SOURCE
}

# Append at the end (after cell 29, the current last cell)
nb["cells"].append(header_cell)
nb["cells"].append(diag_cell)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

total = len(nb["cells"])
print(f"Done. Notebook now has {total} cells.")
print(f"Granularity diagnosis cells inserted at positions {total-2} and {total-1}.")
