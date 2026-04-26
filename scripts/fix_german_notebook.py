"""
Complete the 02_agentic_retrieval_baseline_german.ipynb notebook:
  1. Add LanceDB + embed_model loading cell (after cell 8)
  2. Add tool classes cell (after cell 9 markdown)
  3. Add tool instantiation + TOOLS dict (after LLM loads, cell 11)
  4. Add agent system prompt + run_agent (after cell 15 markdown)
  5. Fix prediction loop cell (replace broken cell 19)
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

NB_PATH = "notebooks/02_agentic_retrieval_baseline_german.ipynb"
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
    raise ValueError(f"Cell not found: {pattern!r}")

# ─── Cell content definitions ────────────────────────────────────────────────

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
        print("LanceDB not found — dense retrieval disabled (BM25-only mode)")
except ImportError:
    _DENSE_AVAILABLE = False
    _lance_table = _embed_model = None
    print("lancedb/sentence_transformers not installed — BM25-only mode")
'''

TOOLS_CELL = '''\
import re as _re_tools


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
        citation = _re_tools.sub(r"\\s*\\([^)]*\\)", "", citation).strip()
        citation = _re_tools.sub(r"[\\s]*[:\\.\\,;]$", "", citation).strip()
        citation = _re_tools.sub(r"\\s+lit\\.\\s+\\w+.*$", "", citation).strip()
        citation = _re_tools.sub(r"\\s+Ziff\\.\\s+\\w+.*$", "", citation).strip()
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

        # 1. Exact match
        for store in [self._laws, self._courts]:
            if citation in store:
                text = store[citation]
                if len(text) > 200:
                    text = text[:200] + "..."
                self._last_citations = [citation]
                return f"[{citation}]\\n{text}"

        # 2. Law-code-aware prefix match
        _art = _re_tools.match(r"^(Art\\.\\s+\\d+\\w*)\\s+([A-Z]\\w+(?:bis|ter)?)$", citation)
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


class LawSearchTool:
    """BM25 search over Swiss federal laws."""

    name: str = "search_laws"
    description: str = (
        "Search Swiss federal laws by German keywords.\\n"
        "Input: German legal keywords  Output: law citations with excerpts"
    )

    def __init__(self, index, top_k=15, max_excerpt=300):
        self.index = index
        self.top_k = top_k
        self.max_excerpt = max_excerpt
        self._last_results: list[dict] = []

    def __call__(self, query: str) -> str:
        return self.run(query)

    def run(self, query: str) -> str:
        if not query or not query.strip():
            return "Error: empty query."
        self._last_results = self.index.search(query, top_k=self.top_k)
        if not self._last_results:
            return f"No laws found for: {query!r}"
        parts = []
        for doc in self._last_results:
            text = doc.get("text", "")[:self.max_excerpt]
            parts.append(f"- {doc.get('citation','?')}: {text}")
        return "\\n".join(parts)

    def get_last_citations(self) -> list[str]:
        return [d.get("citation", "") for d in self._last_results if d.get("citation")]


class CourtSearchTool:
    """BM25 search over Swiss court decisions."""

    name: str = "search_courts"
    description: str = (
        "Search Swiss Federal Court decisions by German keywords.\\n"
        "Input: German legal keywords  Output: court citations with excerpts"
    )

    def __init__(self, index, top_k=15, max_excerpt=300):
        self.index = index
        self.top_k = top_k
        self.max_excerpt = max_excerpt
        self._last_results: list[dict] = []

    def __call__(self, query: str) -> str:
        return self.run(query)

    def run(self, query: str) -> str:
        if not query or not query.strip():
            return "Error: empty query."
        self._last_results = self.index.search(query, top_k=self.top_k)
        if not self._last_results:
            return f"No court decisions found for: {query!r}"
        parts = []
        for doc in self._last_results:
            text = doc.get("text", "")[:self.max_excerpt]
            parts.append(f"- {doc.get('citation','?')}: {text}")
        return "\\n".join(parts)

    def get_last_citations(self) -> list[str]:
        return [d.get("citation", "") for d in self._last_results if d.get("citation")]


class HyDESearchTool:
    """Dense court search via Hypothetical Document Embeddings + BM25 RRF fusion."""

    name: str = "hyde_search_courts"
    description: str = (
        "Search court decisions by meaning using semantic similarity.\\n"
        "Input: 5-8 German legal keywords (NOT citation strings)\\n"
        "Output: most relevant court decision sections\\n"
        "Use when BM25 search fails or for landmark BGE decisions.\\n"
        "Example: \\"Kollusionsgefahr Untersuchungshaft Verhaeltnismaessigkeit\\""
    )

    def __init__(self, llm, embed_model, lance_table, top_k=20, nprobes=20,
                 hyde_tokens=120, max_excerpt=350, dense_available=True):
        self._llm           = llm
        self._embed         = embed_model
        self._table         = lance_table
        self.top_k          = top_k
        self.nprobes        = nprobes
        self.hyde_tokens    = hyde_tokens
        self.max_excerpt    = max_excerpt
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
        out = self._llm(prompt, max_tokens=self.hyde_tokens,
                        temperature=0.3, echo=False)
        return out["choices"][0]["text"].strip()

    def run(self, query: str) -> str:
        query = query.strip().splitlines()[0].strip()
        if not query:
            return "Error: empty query."
        self._last_citations = []

        if self.dense_available and self._embed and self._table:
            hyp = self._generate_hypothetical(query)
            vec = self._embed.encode(
                ["query: " + hyp], normalize_embeddings=True,
                show_progress_bar=False
            )[0].tolist()
            df = (self._table.search(vec).nprobes(self.nprobes)
                  .limit(self.top_k).to_pandas())
            dense = df.to_dict("records") if not df.empty else []
        else:
            hyp   = query
            dense = []

        # BM25 fallback / fusion
        bm25 = courts_index.search(query, top_k=self.top_k)
        fused = reciprocal_rank_fusion(bm25, dense, rrf_k=60)[:self.top_k]

        # Distance filter (dense-only columns)
        DIST_THRESHOLD = 0.35
        top = []
        for doc in fused:
            d = doc.get("_distance", 0.0)
            if "_distance" not in doc or d < DIST_THRESHOLD:
                top.append(doc)
        top = top[:5] or fused[:3]

        self._last_citations = [c.get("citation", "") for c in top]
        parts = []
        for c in top:
            text = c.get("text", "")[:self.max_excerpt]
            parts.append(f"[CITATION_KEY: {c.get('citation','?')}]\\n{text}")

        header = f"Found {len(top)} relevant court sections"
        if self.dense_available:
            header += f". Hypothetical: \\"{hyp[:60]}...\\"\\n"
        return header + "\\n\\n" + "\\n\\n".join(parts)

    def get_last_citations(self) -> list[str]:
        return list(self._last_citations)


# Instantiate BM25 tools (LLM-independent)
law_tool   = LawSearchTool(laws_index,   top_k=CONFIG["top_k_laws"])
court_tool = CourtSearchTool(courts_index, top_k=CONFIG["top_k_courts"])
lookup_tool = CitationLookupTool(
    laws_lut=laws_lookup, courts_lut=courts_lookup, max_prefix_results=3
)
print("BM25 tools ready. HyDE tool will be instantiated after LLM loads.")
'''

HYDE_INST_CELL = '''\
# Instantiate HyDE tool now that LLM is loaded
hyde_tool = HyDESearchTool(
    llm            = llm,
    embed_model    = _embed_model,
    lance_table    = _lance_table,
    top_k          = CONFIG.get("top_k_hyde", 20),
    nprobes        = CONFIG.get("nprobes", 20),
    hyde_tokens    = CONFIG.get("hyde_max_tokens", 120),
    max_excerpt    = CONFIG.get("max_observation_chars", 1200),
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

AGENT_CELL = '''\
import re

AGENT_SYSTEM_PROMPT = """You are a Swiss legal research assistant. Your ONLY job is to find citations.

MANDATORY PROCEDURE - follow in ORDER:

STEP 1 - IDENTIFY DOMAIN
PRE-TRIAL DETENTION (StPO): Art. 221 Abs. 1 StPO, Art. 221 Abs. 2 StPO, Art. 212 Abs. 3 StPO, Art. 222 StPO, Art. 227 Abs. 1 StPO, Art. 393 Abs. 1 StPO, Art. 382 Abs. 1 StPO, Art. 385 Abs. 1 StPO, Art. 390 Abs. 2 StPO, Art. 396 Abs. 1 StPO, Art. 428 Abs. 1 StPO, Art. 422 Abs. 1 StPO, Art. 135 Abs. 3 StPO, Art. 100 Abs. 1 BGG, Art. 37 Abs. 1 StBOG, Art. 39 Abs. 1 StBOG
INVALIDITY (IVG/ATSG): Art. 8 Abs. 1 IVG, Art. 17 Abs. 1 IVG, Art. 28 Abs. 1 IVG, Art. 29 Abs. 1 IVG, Art. 4 Abs. 1 IVG, Art. 8 Abs. 3 IVG, Art. 69 Abs. 1 IVG, Art. 6 ATSG, Art. 8 Abs. 1 ATSG, Art. 16 ATSG, Art. 21 Abs. 4 ATSG, Art. 56 Abs. 1 ATSG, Art. 60 Abs. 1 ATSG, Art. 61 ATSG, Art. 82 BGG, Art. 100 Abs. 1 BGG

STEP 2 - LOOKUP EACH ARTICLE
Call lookup_citation for EACH article from STEP 1. Use the EXACT key with Abs. number.
If lookup returns a CITATION_KEY list, immediately call lookup_citation again with the exact key shown.
CORRECT: lookup_citation("Art. 221 Abs. 1 StPO")
WRONG:   lookup_citation("Art. 221 StPO lit. b\\n\\n2. Art. 212...")
NEVER output "Art. 221 StPO" — always use full key e.g. "Art. 221 Abs. 1 StPO"

STEP 3 - DENSE SEARCH (always do this, German keywords only)
Action: hyde_search_courts
Action Input: [5-8 German legal terms — NO citations, NO English, NO French]
StPO example:     Kollusionsgefahr Untersuchungshaft Verhaeltnismaessigkeit Verdunkelungsgefahr
IVG/ATSG example: Invaliditaetsbemessung Arbeitsfaehigkeit Gutachten Eingliederung IV-Stelle
Then call lookup_citation for each CITATION_KEY returned.

STEP 4 - BM25 SEARCH (German only)
Action: search_courts
Action Input: [German keywords, NOT citation strings]
Then call lookup_citation for each BGE/docket returned.

STEP 5 - OUTPUT
CITATIONS: Art. 221 Abs. 1 StPO; BGE 137 IV 122 E. 6.2; ...
Only emit citations whose text you retrieved via lookup_citation.

RULES:
- Call lookup_citation at least 6 times (always use exact Abs. keys)
- Call hyde_search_courts exactly once with German keywords
- Call search_courts exactly once with German keywords
- Never emit a citation without having called lookup_citation for it
"""


def parse_all_agent_actions(response: str) -> list[tuple[str, str]]:
    actions = []
    action_pattern = r"Action:\\s*(\\w+)"
    input_pattern  = r"Action Input:\\s*(.+?)(?=\\nAction:|$)"
    action_matches = list(re.finditer(action_pattern, response, re.IGNORECASE))
    for i, am in enumerate(action_matches):
        action    = am.group(1).strip()
        start_pos = am.end()
        end_pos   = action_matches[i+1].start() if i+1 < len(action_matches) else len(response)
        m = re.search(input_pattern, response[start_pos:end_pos], re.IGNORECASE | re.DOTALL)
        if m:
            actions.append((action, m.group(1).strip()))
    return actions


def extract_citations_from_text(text: str) -> list[str]:
    found = []
    found += re.findall(r"BGE\\s+\\d{1,3}\\s+[IVX]+[a-z]?\\s+\\d+(?:\\s+E\\.\\s*[\\d\\.]+)?", text)
    found += re.findall(r"Art\\.?\\s+\\d+[a-z]?(?:\\s+Abs\\.?\\s*\\d+)?\\s+[A-Z]{2,}", text)
    found += re.findall(r"\\d+[A-Z]_\\d+/\\d{4}(?:\\s+E\\.\\s*[\\d\\.]+)?", text)
    return list(set(found))


def truncate_observation(obs: str, max_chars: int = 1000) -> str:
    if len(obs) <= max_chars:
        return obs
    return obs[:max_chars] + f"\\n...(truncated, {len(obs)-max_chars} chars)"


def truncate_conversation(conv: str, max_chars: int = 28000) -> str:
    if len(conv) <= max_chars:
        return conv
    inst_end = conv.find("[/INST]")
    if inst_end == -1:
        return "..." + conv[-max_chars:]
    sys_part = conv[:inst_end+7]
    rest     = conv[inst_end+7:]
    budget   = max_chars - len(sys_part) - 100
    if budget <= 0:
        return conv[-max_chars:]
    if len(rest) > budget:
        rest = "\\n...[truncated]...\\n" + rest[-budget:]
    return sys_part + rest


def run_agent(query: str, verbose: bool = False) -> tuple[list[str], list[dict]]:
    conversation = f"[INST] {AGENT_SYSTEM_PROMPT}\\n\\nQuery: {query}\\n\\nThought: [/INST]"
    all_citations: list[str] = []
    logs: list[dict] = []

    for iteration in range(CONFIG["max_iterations"]):
        conversation = truncate_conversation(conversation, CONFIG.get("max_conversation_chars", 28000))

        try:
            response = llm(
                conversation,
                max_tokens=CONFIG["max_tokens"],
                temperature=CONFIG["temperature"],
                stop=["Observation:", "[INST]", "</s>"],
            )["choices"][0]["text"]
        except ValueError as e:
            if "exceed context" in str(e).lower() or "requested tokens" in str(e).lower():
                conversation = truncate_conversation(conversation, 20000)
                try:
                    response = llm(
                        conversation,
                        max_tokens=CONFIG["max_tokens"],
                        temperature=CONFIG["temperature"],
                        stop=["Observation:", "[INST]", "</s>"],
                    )["choices"][0]["text"]
                except ValueError:
                    break
            else:
                raise

        if iteration == 0:
            conversation = (
                f"[INST] {AGENT_SYSTEM_PROMPT}\\n\\nQuery: {query} [/INST]\\n\\nThought:{response}"
            )
        else:
            conversation += response

        logs.append({"type": "llm", "iteration": iteration+1, "response": response[:500]})

        actions = parse_all_agent_actions(response)
        observations = []

        for action, action_input in actions:
            al = action.lower()
            if al in TOOLS:
                tool = TOOLS[al]
                obs  = tool(action_input)
                cits = tool.get_last_citations()
                all_citations.extend(cits)
                obs_trunc = truncate_observation(obs, CONFIG.get("max_observation_chars", 1000))
                observations.append(
                    f"Observation: [{al}({action_input[:40]})] {obs_trunc}"
                )
                logs.append({"type": "tool", "iteration": iteration+1,
                             "tool": action, "query": action_input,
                             "citations": cits, "obs": obs[:300]})
                if verbose:
                    print(f"  [{action}] -> {len(cits)} citations")
            else:
                observations.append(
                    f"Observation: Unknown tool {action!r}. "
                    f"Available: {', '.join(TOOLS.keys())}"
                )

        if observations:
            conversation += "\\n" + "\\n".join(observations) + "\\nThought:"

        if "CITATIONS:" in response:
            final = response.split("CITATIONS:")[-1].strip()
            all_citations += [c.strip() for c in final.split(";") if c.strip()]
            break

        if not actions and "CITATIONS:" not in response:
            all_citations += extract_citations_from_text(response)
            break

    return list(set(all_citations)), logs


print("Agent defined. max_iterations =", CONFIG["max_iterations"])
'''

PRED_CELL = '''\
from tqdm import tqdm

predictions = []
all_logs    = []

QUERY_FILE_PATH = QUERY_FILE  # set in cell 2

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running agent"):
    qid   = row["query_id"]
    query = row["query"]

    citations, logs = run_agent(query, verbose=False)

    predictions.append({
        "query_id":            qid,
        "predicted_citations": ";".join(citations),
    })
    all_logs.append({"query_id": qid, "query": query, "logs": logs})

predictions_df = pd.DataFrame(predictions)
print(f"Done — {len(predictions_df)} predictions")
predictions_df.head()
'''

# ─── Apply changes ───────────────────────────────────────────────────────────

cells = nb["cells"]

# 1. Insert LanceDB cell after cell 8
idx_lancedb = find_cell("Citation lookup dicts") + 1
cells.insert(idx_lancedb, code_cell(LANCEDB_CELL))
print(f"Inserted LanceDB cell at index {idx_lancedb}")

# 2. Insert tool classes after the "## 3. Define Search Tools" markdown
#    (now shifted by 1 from the insert above)
idx_tools_md = find_cell("## 3. Define Search Tools")
cells.insert(idx_tools_md + 1, code_cell(TOOLS_CELL))
print(f"Inserted tools cell at index {idx_tools_md+1}")

# 3. Insert HyDE instantiation after the LLM loading cell
idx_llm = find_cell("from llama_cpp import Llama")
cells.insert(idx_llm + 1, code_cell(HYDE_INST_CELL))
print(f"Inserted TOOLS instantiation at index {idx_llm+1}")

# 4. Insert agent after "## 5. Define ReAct Agent" markdown
idx_agent_md = find_cell("## 5. Define ReAct Agent")
cells.insert(idx_agent_md + 1, code_cell(AGENT_CELL))
print(f"Inserted agent cell at index {idx_agent_md+1}")

# 5. Fix prediction cell — find the broken preview cell and replace it
#    with a complete prediction loop + preview
idx_pred_broken = find_cell("# Preview predictions")
cells[idx_pred_broken]["source"] = PRED_CELL
print(f"Replaced broken prediction cell at index {idx_pred_broken}")

# 6. Add top_k_hyde / nprobes / hyde_max_tokens to CONFIG if missing
idx_cfg = find_cell('"max_iterations"')
cfg_src = src(cells[idx_cfg])
if '"top_k_hyde"' not in cfg_src:
    cfg_src = cfg_src.replace(
        '"top_k_courts": 15,     # Results per court search',
        '"top_k_courts": 15,     # Results per court search\n'
        '    "top_k_hyde": 20,           # Dense retrieval candidate pool\n'
        '    "nprobes": 20,              # IVF clusters to search\n'
        '    "hyde_max_tokens": 120,     # Hypothetical paragraph tokens',
    )
    cells[idx_cfg]["source"] = cfg_src
    print("Added top_k_hyde / nprobes / hyde_max_tokens to CONFIG")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\nSaved: {NB_PATH}  ({len(nb['cells'])} cells)")

# ─── Verify ──────────────────────────────────────────────────────────────────
nb2 = json.load(open(NB_PATH, encoding="utf-8"))
all_src = "".join(
    ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
    for c in nb2["cells"]
)

checks = [
    ("LanceDB loading",          "_DENSE_AVAILABLE" in all_src),
    ("CitationLookupTool",       "class CitationLookupTool" in all_src),
    ("LawSearchTool",            "class LawSearchTool" in all_src),
    ("CourtSearchTool",          "class CourtSearchTool" in all_src),
    ("HyDESearchTool",           "class HyDESearchTool" in all_src),
    ("TOOLS dict",               '"hyde_search_courts": hyde_tool' in all_src),
    ("run_agent defined",        "def run_agent(" in all_src),
    ("agent system prompt",      "MANDATORY PROCEDURE" in all_src),
    ("domain hints StPO",        "Art. 221 Abs. 1 StPO" in all_src),
    ("domain hints IVG",         "Art. 17 Abs. 1 IVG" in all_src),
    ("prediction loop",          'desc="Running agent"' in all_src),
    ("CONFIG top_k_hyde",        '"top_k_hyde"' in all_src),
    ("fr_to_de converter",       '"LAI": "IVG"' in all_src),
    ("prefix keys-only",         "CITATION_KEY: {k}" in all_src),
    ("exact match cap 200",      "if len(text) > 200:" in all_src),
]
print("\nVerification:")
for label, ok in checks:
    print(f"  {'OK' if ok else 'FAIL'} {label}")
