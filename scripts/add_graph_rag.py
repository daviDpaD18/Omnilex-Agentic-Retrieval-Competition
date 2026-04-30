"""
Notebook patch: add GraphRAG support.

1. Insert a new code cell after cell 9 (dense retrieval) that loads
   citation_graph.json into CITATION_GRAPH.
2. Replace Step 3.7 in run_dag_pipeline: swap the free-recall regex scan
   (which only sees the 400-char chunk) with a full prebuilt graph lookup,
   keeping regex as a fallback for decisions not yet in the graph.
"""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")

NB_PATH = "notebooks/02_agentic_retrieval_baseline_german-2.ipynb"
nb = json.load(open(NB_PATH, encoding="utf-8"))

def gs(c):
    s = c["source"]
    return "".join(s) if isinstance(s, list) else s

def make_code_cell(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

# ── 1. Find insertion point (after dense-retrieval cell 9) ───────────────────
dense_idx = next(
    i for i, c in enumerate(nb["cells"])
    if "Dense retrieval components" in gs(c)
)
print(f"Dense retrieval cell: {dense_idx}")

# Guard: skip if already inserted
already_loaded = any("CITATION_GRAPH" in gs(c) for c in nb["cells"])

if already_loaded:
    print("Graph-load cell already present — skipping insertion")
else:
    graph_load_src = (
        "# ── Citation Graph: maps base decision ID → statutes it cites ──────────────\n"
        "# Build once with:  python scripts/build_citation_graph.py  (~5-10 min)\n"
        "import json as _cg_json\n"
        "\n"
        "_GRAPH_PATH = INDEX_PATH / \"citation_graph.json\"\n"
        "\n"
        "if _GRAPH_PATH.exists():\n"
        "    with open(_GRAPH_PATH, encoding=\"utf-8\") as _cg_f:\n"
        "        CITATION_GRAPH: dict[str, list[str]] = _cg_json.load(_cg_f)\n"
        "    print(f\"Citation graph loaded: {len(CITATION_GRAPH):,} decisions\")\n"
        "else:\n"
        "    CITATION_GRAPH = {}\n"
        "    print(\"Citation graph not found — graph-walk will use regex fallback only.\")\n"
        "    print(\"  Build with: python scripts/build_citation_graph.py\")\n"
    )
    nb["cells"].insert(dense_idx + 1, make_code_cell(graph_load_src))
    print(f"Graph-load cell inserted at index {dense_idx + 1}")

# ── 2. Replace Step 3.7 in run_dag_pipeline ──────────────────────────────────
dag_idx = next(i for i, c in enumerate(nb["cells"])
               if "def run_dag_pipeline(" in gs(c))
print(f"DAG pipeline cell: {dag_idx}")

src = gs(nb["cells"][dag_idx])

# Exact old Step 3.7 block (free-recall regex scan of the 400-char chunk)
OLD_37 = (
    "    # ── Step 3.7: Citation Graph Extraction (Free Recall) ────────────────────\n"
    "    # This regex accurately captures Swiss law structures (e.g. Art. 221 Abs. 1 StPO)\n"
    "    art_pattern_ext = r\"Art\\.\\s+\\d+[a-z]?\\s+(?:(?:Abs\\.|Ziff\\.)\\s+\\d+\\w*(?:\\s+lit\\.\\s+\\w+)?\\s+)?[A-Z]\\w+\"\n"
    "    \n"
    "    # Scan the top 5 semantically reranked court decisions for laws they mention\n"
    "    for doc in fused_courts[:5]:\n"
    "        text = doc.get(\"text\", \"\")\n"
    "        extracted_laws = re.findall(art_pattern_ext, text)\n"
    "        for law in extracted_laws:\n"
    "            # Strip trailing whitespace/punctuation just in case\n"
    "            clean_law = law.strip().rstrip(\".,;\")\n"
    "            if clean_law not in statute_citations:\n"
    "                statute_citations.append(clean_law)"
)

NEW_37 = (
    "    # ── Step 3.7: Citation Graph Walk ────────────────────────────────────────\n"
    "    # Look up each top court decision in the prebuilt graph to get ALL statutes\n"
    "    # cited anywhere in that decision (not just the 400-char retrieved chunk).\n"
    "    # Falls back to regex scan of the retrieved chunk for decisions not in graph.\n"
    "    _art_fallback_pat = re.compile(\n"
    "        r\"Art\\.\\s+\\d+[a-z]?\\s+(?:(?:Abs\\.|Ziff\\.)\\s+\\d+\\w*(?:\\s+lit\\.\\s+\\w+)?\\s+)?[A-Z]\\w+\"\n"
    "    )\n"
    "    _graph_hits = 0\n"
    "    for doc in fused_courts[:10]:\n"
    "        cit = doc.get(\"citation\", \"\")\n"
    "        if not cit:\n"
    "            continue\n"
    "        base = cit.split(\" E. \")[0] if \" E. \" in cit else cit\n"
    "        linked = CITATION_GRAPH.get(cit) or CITATION_GRAPH.get(base) or []\n"
    "        if linked:\n"
    "            _graph_hits += 1\n"
    "            for statute in linked:\n"
    "                if statute not in statute_citations:\n"
    "                    statute_citations.append(statute)\n"
    "        else:\n"
    "            # Fallback: regex scan of the retrieved text chunk\n"
    "            for m in _art_fallback_pat.finditer(doc.get(\"text\", \"\")):\n"
    "                clean = m.group(0).strip().rstrip(\".,;\")\n"
    "                if clean not in statute_citations:\n"
    "                    statute_citations.append(clean)"
)

if OLD_37 not in src:
    print("ERROR: Step 3.7 old block not found — check exact whitespace")
    # Show the current 3.7 for debugging
    idx = src.find("Step 3.7")
    print(repr(src[idx:idx+600]))
    sys.exit(1)

src = src.replace(OLD_37, NEW_37, 1)
nb["cells"][dag_idx]["source"] = src
print("Step 3.7 replaced with graph-walk version")

# ── Save ─────────────────────────────────────────────────────────────────────
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("\nNotebook saved.")

# ── Verify ───────────────────────────────────────────────────────────────────
nb2 = json.load(open(NB_PATH, encoding="utf-8"))
s_graph = next(gs(c) for c in nb2["cells"] if "CITATION_GRAPH" in gs(c))
s_dag   = next(gs(c) for c in nb2["cells"] if "def run_dag_pipeline(" in gs(c))

checks = {
    "CITATION_GRAPH load cell present":         "CITATION_GRAPH" in s_graph,
    "graph JSON path uses INDEX_PATH":          "INDEX_PATH" in s_graph,
    "graceful fallback when file missing":      'CITATION_GRAPH = {}' in s_graph,
    "build script hint in load cell":          "build_citation_graph.py" in s_graph,
    "Step 3.7 now says Citation Graph Walk":    "Citation Graph Walk" in s_dag,
    "CITATION_GRAPH.get lookup in step 3.7":    "CITATION_GRAPH.get(" in s_dag,
    "base_id extraction from ' E. '":           '" E. "' in s_dag,
    "top 10 court decisions walked":            "fused_courts[:10]" in s_dag,
    "regex fallback present":                   "_art_fallback_pat" in s_dag,
    "old free-recall comment gone":             "Free Recall" not in s_dag,
}
all_ok = True
print()
for k, v in checks.items():
    print(f"  {'OK' if v else 'FAIL'} {k}")
    if not v:
        all_ok = False
print()
print("All checks passed!" if all_ok else "Some checks FAILED.")
