"""
Apply all improvements to 02_agentic_retrieval_baseline.ipynb:
  1. CONFIG: add top_k_hyde/nprobes/hyde_max_tokens, max_iterations->7
  2. Insert LanceDB + embed_model loading cell after cell 8
  3. Replace CitationLookupTool with improved version (_normalize, _fr_to_de,
     law-code-aware prefix, keys-only prefix, 200-char exact cap)
  4. Add HyDESearchTool class + update TOOLS instantiation
  5. Add hyde_tool instantiation cell after LLM loads
  6. Replace AGENT_SYSTEM_PROMPT with new 5-step domain-hint version
"""
import json, sys, re
sys.stdout.reconfigure(encoding='utf-8')

NB_PATH = "notebooks/02_agentic_retrieval_baseline.ipynb"
nb = json.load(open(NB_PATH, encoding='utf-8'))

def code_cell(source):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source}

def src(c):
    s = c["source"]
    return "".join(s) if isinstance(s, list) else s

def find_cell(pattern):
    for i, c in enumerate(nb['cells']):
        if pattern in src(c):
            return i
    raise ValueError(f"Pattern not found: {pattern!r}")

cells = nb["cells"]

# ── 1. CONFIG updates ─────────────────────────────────────────────────────────
idx_cfg = find_cell('"max_iterations"')
s = src(cells[idx_cfg])

if '"top_k_hyde"' not in s:
    s = s.replace(
        '"top_k_courts": 15,     # Results per court search',
        '"top_k_courts": 15,     # Results per court search\n'
        '    "top_k_hyde": 20,           # Dense retrieval candidate pool\n'
        '    "nprobes": 20,              # IVF clusters to search\n'
        '    "hyde_max_tokens": 120,     # Hypothetical paragraph tokens',
    )

s = re.sub(r'"max_iterations":\s*\d+,', '"max_iterations": 7,', s)
cells[idx_cfg]["source"] = s
print("1. CONFIG updated (top_k_hyde, nprobes, hyde_max_tokens, max_iterations=7)")

# ── 2. LanceDB loading cell after citation lookup dicts ───────────────────────
LANCEDB_CELL = '''\
# ── Dense retrieval components (optional — degrades to BM25-only if unavailable)
import torch

try:
    import lancedb
    from sentence_transformers import SentenceTransformer

    if KAGGLE_ENV:
        LANCEDB_PATH = Path("/kaggle/input/omnilex-indices/lancedb_courts")
    else:
        LANCEDB_PATH = REPO_ROOT / "data" / "processed" / "lancedb_courts"

    if LANCEDB_PATH.exists():
        _lance_db    = lancedb.connect(str(LANCEDB_PATH))
        _lance_table = _lance_db.open_table("courts")
        _device      = "cuda" if torch.cuda.is_available() else "cpu"
        _embed_model = SentenceTransformer(
            "intfloat/multilingual-e5-base", device=_device
        )
        _embed_model.max_seq_length = 512
        _DENSE_AVAILABLE = True
        print(f"Dense retrieval ready  (device={_device})")
    else:
        _DENSE_AVAILABLE = False
        _lance_table = _embed_model = None
        print("LanceDB not found — BM25-only mode")
except ImportError:
    _DENSE_AVAILABLE = False
    _lance_table = _embed_model = None
    print("lancedb/sentence_transformers not installed — BM25-only mode")
'''

idx_lut = find_cell("Citation lookup dicts")
cells.insert(idx_lut + 1, code_cell(LANCEDB_CELL))
print(f"2. LanceDB cell inserted at index {idx_lut + 1}")

# ── 3+4. Replace CitationLookupTool + add HyDESearchTool in tools cell ────────
# Index has shifted by 1
idx_tools = find_cell("class CitationLookupTool")

OLD_LOOKUP_TOOL = src(cells[idx_tools])

# Find where CitationLookupTool starts and where TOOLS dict ends
clt_start = OLD_LOOKUP_TOOL.index("class CitationLookupTool")
# Find end: "TOOLS = {" block + closing "}"
tools_dict_end = OLD_LOOKUP_TOOL.rfind("}")
suffix_after_tools = OLD_LOOKUP_TOOL[tools_dict_end + 1:]  # anything after last }

prefix_before_clt = OLD_LOOKUP_TOOL[:clt_start]

NEW_CLT_AND_HYDE = '''\
class CitationLookupTool:
    """O(1) direct lookup by citation key with normalization and prefix matching."""

    name: str = "lookup_citation"
    description: str = (
        "Look up the exact text of a specific legal citation.\\n"
        "Input: citation string e.g. \\"Art. 221 Abs. 1 StPO\\" or \\"BGE 137 IV 122 E. 6.2\\"\\n"
        "Output: text of that provision, or a list of matching keys for prefix searches.\\n"
        "Use this when you know which article or decision is likely relevant."
    )
    _fr_to_de = {
        "LAI": "IVG", "CO": "OR", "CC": "ZGB", "CP": "StGB",
        "CPP": "StPO", "LPGA": "ATSG", "LTF": "BGG",
        "LAA": "UVG", "LAVS": "AHVG", "LPP": "BVG",
    }

    def __init__(self, laws_lut, courts_lut, max_prefix_results=3):
        self._laws   = laws_lut
        self._courts = courts_lut
        self.max_prefix_results = max_prefix_results
        self._last_citations: list[str] = []

    def __call__(self, citation: str) -> str:
        return self.run(citation)

    def _normalize(self, citation: str) -> str:
        for stop in ["\\n", "Thought:", "Action:", "Now ", "Found ", "Looking", "Next"]:
            if stop in citation:
                citation = citation[:citation.index(stop)]
        citation = citation.strip()
        citation = re.sub(r"\\s*\\([^)]*\\)", "", citation).strip()
        citation = re.sub(r"[\\s]*[:\\.\\,;]$", "", citation).strip()
        citation = re.sub(r"\\s+lit\\.\\s+\\w+.*$", "", citation).strip()
        citation = re.sub(r"\\s+Ziff\\.\\s+\\w+.*$", "", citation).strip()
        tokens = citation.split()
        if tokens and tokens[-1] in self._fr_to_de:
            tokens[-1] = self._fr_to_de[tokens[-1]]
            citation = " ".join(tokens)
        return citation

    def run(self, citation: str) -> str:
        citation = self._normalize(citation)
        if not citation:
            return "Error: empty citation string."
        self._last_citations = []

        # 1. Exact match — cap at 200 chars
        for store in [self._laws, self._courts]:
            if citation in store:
                text = store[citation]
                if len(text) > 200:
                    text = text[:200] + "..."
                self._last_citations = [citation]
                return f"[{citation}]\\n{text}"

        # 2. Law-code-aware prefix match
        _art = re.match(r"^(Art\\.\\s+\\d+\\w*)\\s+([A-Z]\\w+(?:bis|ter)?)$", citation)
        candidates = []
        for store in [self._laws, self._courts]:
            if _art:
                pfx, sfx = _art.group(1), _art.group(2)
                candidates += [(k, v) for k, v in store.items()
                               if k.startswith(pfx) and sfx in k]
            else:
                candidates += [(k, v) for k, v in store.items()
                               if k.startswith(citation)]

        if candidates:
            self._last_citations = [k for k, _ in candidates[:self.max_prefix_results]]
            key_list = "\\n".join(
                f"  CITATION_KEY: {k}" for k, _ in candidates[:self.max_prefix_results]
            )
            return (
                f"Prefix \\"{citation}\\" matched {len(candidates)} citations. "
                f"Top {min(len(candidates), self.max_prefix_results)} keys:\\n{key_list}\\n\\n"
                f"Call lookup_citation with the EXACT key, e.g. lookup_citation(\\"Art. 100 Abs. 1 BGG\\")"
            )

        return (
            f"Citation \\"{citation}\\" not found. "
            f"Valid formats: \\"Art. 221 Abs. 1 StPO\\", \\"BGE 137 IV 122 E. 6.2\\", \\"1B_90/2021 E. 2.1\\""
        )

    def get_last_citations(self) -> list[str]:
        return list(self._last_citations)


class HyDESearchTool:
    """Dense court search via Hypothetical Document Embeddings + BM25 RRF fusion."""

    name: str = "hyde_search_courts"
    description: str = (
        "Search court decisions by meaning using semantic similarity.\\n"
        "Input: 5-8 German legal keywords (NOT citation strings)\\n"
        "Output: most relevant court decision sections\\n"
        "Example: \\"Kollusionsgefahr Untersuchungshaft Verhaeltnismaessigkeit\\""
    )

    def __init__(self, llm, embed_model, lance_table, top_k=20, nprobes=20,
                 hyde_tokens=120, max_excerpt=350, dense_available=True):
        self._llm            = llm
        self._embed          = embed_model
        self._table          = lance_table
        self.top_k           = top_k
        self.nprobes         = nprobes
        self.hyde_tokens     = hyde_tokens
        self.max_excerpt     = max_excerpt
        self.dense_available = dense_available
        self._last_citations: list[str] = []

    def __call__(self, query: str) -> str:
        return self.run(query)

    def _generate_hypothetical(self, query: str) -> str:
        prompt = (
            "[INST] Du bist ein Schweizer Bundesrichter. "
            "Schreibe einen Erwaegungsabsatz (ca. 80 Woerter) auf Deutsch "
            "der die folgende Rechtsfrage beantwortet. "
            "Nur der Erwaegungstext, keine Einleitung.\\n\\n"
            f"Rechtsfrage: {query} [/INST]"
        )
        out = self._llm(prompt, max_tokens=self.hyde_tokens, temperature=0.3, echo=False)
        return out["choices"][0]["text"].strip()

    def run(self, query: str) -> str:
        query = query.strip().splitlines()[0].strip()
        if not query:
            return "Error: empty query."
        self._last_citations = []

        dense = []
        if self.dense_available and self._embed and self._table:
            hyp = self._generate_hypothetical(query)
            vec = self._embed.encode(
                ["query: " + hyp], normalize_embeddings=True, show_progress_bar=False
            )[0].tolist()
            df = (self._table.search(vec).nprobes(self.nprobes)
                  .limit(self.top_k).to_pandas())
            dense = df.to_dict("records") if not df.empty else []
        else:
            hyp = query

        # BM25 fusion
        bm25 = courts_index.search(query, top_k=self.top_k)
        fused = _rrf_merge(bm25, dense)[:self.top_k]

        DIST_THRESHOLD = 0.35
        top = [d for d in fused if d.get("_distance", 0.0) < DIST_THRESHOLD
               or "_distance" not in d]
        top = top[:5] or fused[:3]

        self._last_citations = [c.get("citation", "") for c in top]
        parts = [
            f"[CITATION_KEY: {c.get('citation','?')}]\\n{c.get('text','')[:self.max_excerpt]}"
            for c in top
        ]
        return f"Found {len(top)} relevant court sections.\\n\\n" + "\\n\\n".join(parts)

    def get_last_citations(self) -> list[str]:
        return list(self._last_citations)


def _rrf_merge(bm25_results, dense_results, k=60):
    scores, store = {}, {}
    for rank, doc in enumerate(bm25_results, 1):
        cit = doc.get("citation")
        if not cit: continue
        store[cit] = doc
        scores[cit] = scores.get(cit, 0.0) + 1.0 / (k + rank)
    for rank, doc in enumerate(dense_results, 1):
        cit = doc.get("citation")
        if not cit: continue
        if cit not in store: store[cit] = doc
        scores[cit] = scores.get(cit, 0.0) + 1.0 / (k + rank)
    return [store[c] for c in sorted(scores, key=lambda x: scores[x], reverse=True)]


# Instantiate BM25 tools (LLM-independent)
law_tool = LawSearchTool(
    index=laws_index, top_k=CONFIG["top_k_laws"], max_excerpt_length=300,
)
court_tool = CourtSearchTool(
    index=courts_index, top_k=CONFIG["top_k_courts"], max_excerpt_length=300,
)
lookup_tool = CitationLookupTool(
    laws_lut=laws_lookup, courts_lut=courts_lookup, max_prefix_results=3,
)
print("BM25 tools ready. HyDE tool will be instantiated after LLM loads.")
'''

cells[idx_tools]["source"] = prefix_before_clt + NEW_CLT_AND_HYDE
print(f"3+4. CitationLookupTool replaced + HyDESearchTool added in cell {idx_tools}")

# ── 5. Insert hyde_tool instantiation after LLM cell ─────────────────────────
HYDE_INST_CELL = '''\
# Instantiate HyDE tool now that LLM is loaded, update TOOLS dict
hyde_tool = HyDESearchTool(
    llm             = llm,
    embed_model     = _embed_model,
    lance_table     = _lance_table,
    top_k           = CONFIG.get("top_k_hyde", 20),
    nprobes         = CONFIG.get("nprobes", 20),
    hyde_tokens     = CONFIG.get("hyde_max_tokens", 120),
    max_excerpt     = CONFIG.get("max_observation_chars", 1000),
    dense_available = _DENSE_AVAILABLE,
)

TOOLS = {
    "search_laws":        law_tool,
    "search_courts":      court_tool,
    "lookup_citation":    lookup_tool,
    "hyde_search_courts": hyde_tool,
}
print("All tools ready:", list(TOOLS.keys()))
'''

idx_llm = find_cell("from llama_cpp import Llama")
cells.insert(idx_llm + 1, code_cell(HYDE_INST_CELL))
print(f"5. Hyde instantiation cell inserted at index {idx_llm + 1}")

# ── 6. Replace AGENT_SYSTEM_PROMPT ───────────────────────────────────────────
idx_agent = find_cell("AGENT_SYSTEM_PROMPT")
s = src(cells[idx_agent])

start_marker = 'AGENT_SYSTEM_PROMPT = """'
prompt_start = s.index(start_marker)
content_start = prompt_start + len(start_marker)
prompt_end = s.index('"""', content_start)

NEW_PROMPT = (
    "You are a Swiss legal research assistant. Your ONLY job is to find citations.\n"
    "\n"
    "MANDATORY PROCEDURE - follow in ORDER:\n"
    "\n"
    "STEP 1 - IDENTIFY DOMAIN\n"
    "PRE-TRIAL DETENTION (StPO): Art. 221 Abs. 1 StPO, Art. 221 Abs. 2 StPO, Art. 212 Abs. 3 StPO, Art. 222 StPO, Art. 227 Abs. 1 StPO, Art. 393 Abs. 1 StPO, Art. 382 Abs. 1 StPO, Art. 385 Abs. 1 StPO, Art. 390 Abs. 2 StPO, Art. 396 Abs. 1 StPO, Art. 428 Abs. 1 StPO, Art. 422 Abs. 1 StPO, Art. 135 Abs. 3 StPO, Art. 100 Abs. 1 BGG, Art. 37 Abs. 1 StBOG, Art. 39 Abs. 1 StBOG\n"
    "INVALIDITY (IVG/ATSG): Art. 8 Abs. 1 IVG, Art. 17 Abs. 1 IVG, Art. 28 Abs. 1 IVG, Art. 29 Abs. 1 IVG, Art. 4 Abs. 1 IVG, Art. 8 Abs. 3 IVG, Art. 69 Abs. 1 IVG, Art. 6 ATSG, Art. 8 Abs. 1 ATSG, Art. 16 ATSG, Art. 21 Abs. 4 ATSG, Art. 56 Abs. 1 ATSG, Art. 60 Abs. 1 ATSG, Art. 61 ATSG, Art. 82 BGG, Art. 100 Abs. 1 BGG\n"
    "\n"
    "STEP 2 - LOOKUP EACH ARTICLE\n"
    "Call lookup_citation for EACH article from STEP 1. Use the EXACT key with Abs. number.\n"
    "If lookup returns a CITATION_KEY list, immediately call lookup_citation again with the exact key shown.\n"
    'CORRECT: lookup_citation("Art. 221 Abs. 1 StPO")\n'
    'WRONG:   lookup_citation("Art. 221 StPO lit. b\\n\\n2. Art. 212...")\n'
    'NEVER output "Art. 221 StPO" — always use full key e.g. "Art. 221 Abs. 1 StPO"\n'
    "\n"
    "STEP 3 - DENSE SEARCH (always do this, German keywords only)\n"
    "Action: hyde_search_courts\n"
    "Action Input: [5-8 German legal terms — NO citations, NO English, NO French]\n"
    "StPO example:     Kollusionsgefahr Untersuchungshaft Verhaeltnismaessigkeit Verdunkelungsgefahr\n"
    "IVG/ATSG example: Invaliditaetsbemessung Arbeitsfaehigkeit Gutachten Eingliederung IV-Stelle\n"
    "Then call lookup_citation for each CITATION_KEY returned.\n"
    "\n"
    "STEP 4 - BM25 SEARCH (German only)\n"
    "Action: search_courts\n"
    "Action Input: [German keywords, NOT citation strings]\n"
    "Then call lookup_citation for each BGE/docket returned.\n"
    "\n"
    "STEP 5 - OUTPUT\n"
    "CITATIONS: Art. 221 Abs. 1 StPO; BGE 137 IV 122 E. 6.2; ...\n"
    "Only emit citations whose text you retrieved via lookup_citation.\n"
    "\n"
    "RULES:\n"
    "- Call lookup_citation at least 6 times (always use exact Abs. keys)\n"
    "- Call hyde_search_courts exactly once with German keywords\n"
    "- Call search_courts exactly once with German keywords\n"
    "- Never emit a citation without having called lookup_citation for it\n"
)

s = s[:prompt_start] + start_marker + NEW_PROMPT + '"""' + s[prompt_end + 3:]
cells[idx_agent]["source"] = s
print(f"6. AGENT_SYSTEM_PROMPT replaced in cell {idx_agent}")

# ── Save ──────────────────────────────────────────────────────────────────────
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\nSaved: {NB_PATH}  ({len(nb['cells'])} cells)")

# ── Verify ────────────────────────────────────────────────────────────────────
nb2 = json.load(open(NB_PATH, encoding="utf-8"))
all_src = "".join(
    ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
    for c in nb2["cells"]
)
print("\nVerification:")
checks = [
    ("max_iterations=7",          '"max_iterations": 7,' in all_src),
    ("top_k_hyde in CONFIG",      '"top_k_hyde"' in all_src),
    ("LanceDB loading",           "_DENSE_AVAILABLE" in all_src),
    ("_normalize method",         "def _normalize(" in all_src),
    ("_fr_to_de converter",       '"LAI": "IVG"' in all_src),
    ("law-code-aware prefix",     "_art.group(1)" in all_src),
    ("prefix keys-only",          "CITATION_KEY: {k}" in all_src),
    ("exact match cap 200",       "if len(text) > 200:" in all_src),
    ("max_prefix_results=3",      "max_prefix_results=3" in all_src),
    ("HyDESearchTool class",      "class HyDESearchTool" in all_src),
    ("_rrf_merge defined",        "def _rrf_merge(" in all_src),
    ("hyde in TOOLS",             '"hyde_search_courts": hyde_tool' in all_src),
    ("5-step prompt",             "MANDATORY PROCEDURE - follow in ORDER" in all_src),
    ("StPO domain hints",         "Art. 221 Abs. 1 StPO" in all_src),
    ("IVG domain hints",          "Art. 17 Abs. 1 IVG" in all_src),
]
for label, ok in checks:
    print(f"  {'OK' if ok else 'FAIL'} {label}")
