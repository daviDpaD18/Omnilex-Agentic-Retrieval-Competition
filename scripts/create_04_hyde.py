import json, re, copy

SRC_PATH = "notebooks/02_agentic_retrieval_baseline.ipynb"
DST_PATH = "notebooks/04_hyde_agent.ipynb"

nb = json.load(open(SRC_PATH, encoding="utf-8"))
nb = copy.deepcopy(nb)

# ── Addition 1: New CONFIG keys after top_k_courts ────────────────────────
config_idx = next(i for i,c in enumerate(nb["cells"]) if '"top_k_courts"' in "".join(c["source"]))
src = "".join(nb["cells"][config_idx]["source"])
OLD_CFG = '"top_k_courts": 15,     # Results per court search'
NEW_CFG = (
    '"top_k_courts": 15,     # Results per court search\n'
    '    "top_k_hyde": 100,          # Dense retrieval candidate pool\n'
    '    "nprobes":    20,            # IVF clusters to search (out of 256)\n'
    '    "hyde_max_tokens": 200,     # Tokens for hypothetical paragraph generation\n'
    '    "reranker_model": "nreimers/mmarco-mMiniLMv2-L12-H384-v1",\n'
    '    "top_k_rerank": 15,         # Final results after reranking'
)
assert OLD_CFG in src, "CONFIG anchor not found"
nb["cells"][config_idx]["source"] = src.replace(OLD_CFG, NEW_CFG, 1)
print("Addition 1 done: CONFIG keys")

# ── Addition 2: LANCEDB_PATH after COURTS_INDEX_PATH ─────────────────────
path_idx = next(i for i,c in enumerate(nb["cells"]) if "COURTS_INDEX_PATH" in "".join(c["source"]))
src = "".join(nb["cells"][path_idx]["source"])
OLD_PATH = 'COURTS_INDEX_PATH = INDEX_PATH / "courts_index.pkl"'
NEW_PATH = (
    'COURTS_INDEX_PATH = INDEX_PATH / "courts_index.pkl"\n'
    "\n"
    "if KAGGLE_ENV:\n"
    '    LANCEDB_PATH = Path("/kaggle/input/omnilex-indices/lancedb_courts")\n'
    "else:\n"
    '    LANCEDB_PATH = REPO_ROOT / "data" / "processed" / "lancedb_courts"'
)
assert OLD_PATH in src, "COURTS_INDEX_PATH anchor not found"
nb["cells"][path_idx]["source"] = src.replace(OLD_PATH, NEW_PATH, 1)
print("Addition 2 done: LANCEDB_PATH")

# ── Addition 3: Dense components cell after courts loading cell ───────────
courts_cell_idx = next(i for i,c in enumerate(nb["cells"]) if "courts_index = get_or_build_index" in "".join(c["source"]))
DENSE_SOURCE = """\
# -- Dense retrieval components for HyDE ---------------------------------
import lancedb
from sentence_transformers import SentenceTransformer, CrossEncoder

# LanceDB -- mmap, no RAM cost, instant load
print("Loading LanceDB courts index...")
_lance_db    = lancedb.connect(str(LANCEDB_PATH))
_lance_table = _lance_db.open_table("courts")
print("  LanceDB ready")

# Embedding model -- loads on GPU alongside Mistral
# VRAM: Mistral 4.5GB + e5-base 0.42GB = 4.92GB, within RTX 4050 budget
print("Loading embedding model...")
_embed_model = SentenceTransformer(
    "intfloat/multilingual-e5-base", device="cuda"
)
_embed_model.max_seq_length = 512
print("  Embedding model ready")

# Cross-encoder reranker -- 120MB VRAM
print("Loading cross-encoder reranker...")
_reranker = CrossEncoder(
    CONFIG["reranker_model"], device="cuda", max_length=512
)
print("  Reranker ready")
"""
nb["cells"].insert(courts_cell_idx + 1, {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": DENSE_SOURCE,
})
print(f"Addition 3 done: dense components cell at index {courts_cell_idx + 1}")

# ── Addition 4+5: HyDESearchTool class + updated TOOLS dict ──────────────
tools_idx = next(i for i,c in enumerate(nb["cells"]) if "class CitationLookupTool" in "".join(c["source"]))
src = "".join(nb["cells"][tools_idx]["source"])

HYDE_CLASS = r'''

class HyDESearchTool:
    """Dense court search using Hypothetical Document Embeddings.

    Generates a fake BGE reasoning paragraph via Mistral, embeds it
    with multilingual-e5-base, searches LanceDB for similar real
    court sections, then reranks with a cross-encoder.

    Use this for finding relevant BGE decisions and recent docket
    citations that BM25 cannot retrieve via natural language queries.
    """

    name: str = "hyde_search_courts"
    description: str = (
        "Search Swiss Federal Court decisions using semantic similarity.\n"
        "Input: A German legal question or concept (NOT a citation string)\n"
        "Output: Most semantically relevant court decision sections\n\n"
        "Use this when search_courts (BM25) fails to find relevant BGE decisions.\n"
        "Example queries (in German):\n"
        "  'Kollusionsgefahr konkrete Gefaehrdung Zeugen Untersuchungshaft'\n"
        "  'Invaliditaetsbemessung Arbeitsfaehigkeit widerspruechliche Gutachten'"
    )

    def __init__(
        self,
        llm,
        embed_model,
        lance_table,
        reranker,
        top_k_dense:  int = 100,
        nprobes:      int = 20,
        top_k_final:  int = 15,
        hyde_tokens:  int = 200,
        max_excerpt:  int = 400,
    ):
        self._llm         = llm
        self._embed       = embed_model
        self._table       = lance_table
        self._reranker    = reranker
        self.top_k_dense  = top_k_dense
        self.nprobes      = nprobes
        self.top_k_final  = top_k_final
        self.hyde_tokens  = hyde_tokens
        self.max_excerpt  = max_excerpt
        self._last_citations: list[str] = []

    def __call__(self, query: str) -> str:
        return self.run(query)

    def _generate_hypothetical(self, query: str) -> str:
        prompt = (
            "[INST] Du bist ein Schweizer Bundesrichter. "
            "Schreibe einen Erwaegungsabsatz (ca. 120 Woerter) auf Deutsch, "
            "der die folgende Rechtsfrage beantwortet. "
            "Verwende typische bundesgerichtliche Formulierungen. "
            "Schreibe NUR den Erwaegungstext, keine Einleitung.\n\n"
            f"Rechtsfrage: {query} [/INST]"
        )
        output = self._llm(
            prompt,
            max_tokens=self.hyde_tokens,
            temperature=0.3,
            echo=False,
        )
        return output["choices"][0]["text"].strip()

    def run(self, query: str) -> str:
        query = query.strip().splitlines()[0].strip()
        if not query:
            return "Error: empty query."

        self._last_citations = []

        hypothetical = self._generate_hypothetical(query)

        vec = self._embed.encode(
            ["query: " + hypothetical],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()

        results_df = (
            self._table.search(vec)
                .nprobes(self.nprobes)
                .limit(self.top_k_dense)
                .to_pandas()
        )
        if results_df.empty:
            return "No results from dense search."

        candidates = results_df.to_dict("records")

        pairs = [[query, c.get("text", "")[:400]] for c in candidates]
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            zip(scores, candidates), key=lambda x: x[0], reverse=True
        )
        top = [c for _, c in ranked[:self.top_k_final]]

        self._last_citations = [c.get("citation", "") for c in top]
        parts = []
        for c in top:
            cit  = c.get("citation", "?")
            text = c.get("text", "")[:self.max_excerpt]
            parts.append(f"[CITATION_KEY: {cit}]\n{text}")

        header = (
            f"Dense search found {len(top)} results. "
            f"Hypothetical used: \"{hypothetical[:80]}...\"\n\n"
        )
        return header + "\n\n".join(parts)

    def get_last_citations(self) -> list[str]:
        return list(self._last_citations)
'''

OLD_TOOLS_BLOCK = (
    'TOOLS = {\n'
    '    "search_laws": law_tool,\n'
    '    "search_courts": court_tool,\n'
    '    "lookup_citation": lookup_tool,\n'
    '}'
)
NEW_TOOLS_BLOCK = (
    'hyde_tool = HyDESearchTool(\n'
    '    llm         = llm,\n'
    '    embed_model = _embed_model,\n'
    '    lance_table = _lance_table,\n'
    '    reranker    = _reranker,\n'
    '    top_k_dense = CONFIG["top_k_hyde"],\n'
    '    nprobes     = CONFIG["nprobes"],\n'
    '    top_k_final = CONFIG["top_k_rerank"],\n'
    '    hyde_tokens = CONFIG["hyde_max_tokens"],\n'
    '    max_excerpt = CONFIG["max_observation_chars"],\n'
    ')\n'
    '\n'
    'TOOLS = {\n'
    '    "search_laws":        law_tool,\n'
    '    "search_courts":      court_tool,\n'
    '    "lookup_citation":    lookup_tool,\n'
    '    "hyde_search_courts": hyde_tool,\n'
    '}'
)
assert OLD_TOOLS_BLOCK in src, f"TOOLS dict anchor not found.\nLooking for:\n{OLD_TOOLS_BLOCK}"
src = src.replace(OLD_TOOLS_BLOCK, HYDE_CLASS + "\n\n" + NEW_TOOLS_BLOCK, 1)
nb["cells"][tools_idx]["source"] = src
print("Addition 4+5 done: HyDESearchTool + updated TOOLS dict")

# ── Addition 6: hyde_search_courts step in AGENT_SYSTEM_PROMPT ───────────
agent_idx = next(i for i,c in enumerate(nb["cells"]) if "AGENT_SYSTEM_PROMPT" in "".join(c["source"]))
src = "".join(nb["cells"][agent_idx]["source"])

STEP4_ANCHOR = "STEP 4 - LOOKUP SPECIFIC DECISIONS"
assert STEP4_ANCHOR in src, "STEP 4 anchor not found in agent prompt"

HYDE_STEP = (
    "STEP 3b - DENSE SEMANTIC COURT SEARCH (use when BM25 search returns\n"
    "irrelevant results or you need landmark BGE decisions)\n\n"
    "Action: hyde_search_courts\n"
    "Action Input: Kollusionsgefahr Untersuchungshaft konkrete Gefaehrdung Zeugen\n\n"
    "This tool finds court decisions by MEANING not keywords.\n"
    "Use it with German legal concepts, NOT citation strings.\n"
    "All CITATION_KEYs returned must be included in your final answer.\n\n"
)
src = src.replace(STEP4_ANCHOR, HYDE_STEP + STEP4_ANCHOR, 1)
nb["cells"][agent_idx]["source"] = src
print("Addition 6 done: hyde_search_courts in agent prompt")

# ── Update title ──────────────────────────────────────────────────────────
title_src = "".join(nb["cells"][0]["source"])
nb["cells"][0]["source"] = title_src.replace(
    "# Agentic Retrieval Baseline",
    "# Agentic Retrieval -- HyDE + Reranker"
).replace("02_agentic", "04_hyde")

# ── Save ──────────────────────────────────────────────────────────────────
with open(DST_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\nSaved: {DST_PATH}  ({len(nb['cells'])} cells)")

# ── Quick verification ────────────────────────────────────────────────────
nb2 = json.load(open(DST_PATH, encoding="utf-8"))
checks = [
    ("top_k_hyde in CONFIG",      '"top_k_hyde"' in "".join(nb2["cells"][config_idx]["source"])),
    ("LANCEDB_PATH defined",      "LANCEDB_PATH" in "".join(nb2["cells"][path_idx]["source"])),
    ("Dense components cell",     "lancedb.connect" in "".join(c["source"] for c in nb2["cells"])),
    ("HyDESearchTool class",      "class HyDESearchTool" in "".join(c["source"] for c in nb2["cells"])),
    ("hyde_tool instantiation",   "hyde_tool = HyDESearchTool" in "".join(c["source"] for c in nb2["cells"])),
    ("hyde_search_courts in TOOLS","hyde_search_courts" in "".join(c["source"] for c in nb2["cells"])),
    ("hyde step in prompt",       "STEP 3b" in "".join(c["source"] for c in nb2["cells"])),
]
print("\nVerification:")
for label, ok in checks:
    print(f"  {'OK' if ok else 'FAIL'} {label}")
