"""Apply 6 targeted changes to 04_hyde_agent.ipynb."""
import json, re

NB_PATH = "notebooks/04_hyde_agent.ipynb"
nb = json.load(open(NB_PATH, encoding="utf-8"))

def src(c):
    s = c["source"]
    return "".join(s) if isinstance(s, list) else s

def set_src(c, s):
    c["source"] = s

results = []

# ══════════════════════════════════════════════════════════════════════════
# Change 1: CONFIG — top_k_hyde, hyde_max_tokens, max_iterations
# ══════════════════════════════════════════════════════════════════════════
config_idx = 3
s = src(nb["cells"][config_idx])
s = s.replace('"top_k_hyde": 100,', '"top_k_hyde": 20,', 1)
s = s.replace('"hyde_max_tokens": 200,', '"hyde_max_tokens": 120,', 1)
s = s.replace('"max_iterations": 8,', '"max_iterations": 6,', 1)
set_src(nb["cells"][config_idx], s)
# Verify
ok = '"top_k_hyde": 20,' in s and '"hyde_max_tokens": 120,' in s and '"max_iterations": 6,' in s
results.append(f"Change 1 {'OK' if ok else 'FAIL'}: CONFIG top_k_hyde=20, hyde_max_tokens=120, max_iterations=6")

# ══════════════════════════════════════════════════════════════════════════
# Change 2: HyDESearchTool — remove cross-encoder, replace __init__ and run()
# ══════════════════════════════════════════════════════════════════════════
tools_idx = 11
s = src(nb["cells"][tools_idx])

OLD_HYDE_INIT = (
    "    def __init__(\n"
    "        self,\n"
    "        llm,\n"
    "        embed_model,\n"
    "        lance_table,\n"
    "        reranker,\n"
    "        top_k_dense:  int = 100,\n"
    "        nprobes:      int = 20,\n"
    "        top_k_final:  int = 15,\n"
    "        hyde_tokens:  int = 200,\n"
    "        max_excerpt:  int = 400,\n"
    "    ):\n"
    "        self._llm         = llm\n"
    "        self._embed       = embed_model\n"
    "        self._table       = lance_table\n"
    "        self._reranker    = reranker\n"
    "        self.top_k_dense  = top_k_dense\n"
    "        self.nprobes      = nprobes\n"
    "        self.top_k_final  = top_k_final\n"
    "        self.hyde_tokens  = hyde_tokens\n"
    "        self.max_excerpt  = max_excerpt\n"
    "        self._last_citations: list[str] = []\n"
)

NEW_HYDE_INIT = (
    "    def __init__(\n"
    "        self,\n"
    "        llm,\n"
    "        embed_model,\n"
    "        lance_table,\n"
    "        top_k_dense: int = 20,\n"
    "        nprobes:     int = 20,\n"
    "        hyde_tokens: int = 120,\n"
    "        max_excerpt: int = 400,\n"
    "    ):\n"
    "        self._llm        = llm\n"
    "        self._embed      = embed_model\n"
    "        self._table      = lance_table\n"
    "        self.top_k_dense = top_k_dense\n"
    "        self.nprobes     = nprobes\n"
    "        self.hyde_tokens = hyde_tokens\n"
    "        self.max_excerpt = max_excerpt\n"
    "        self._last_citations: list[str] = []\n"
)

if OLD_HYDE_INIT in s:
    s = s.replace(OLD_HYDE_INIT, NEW_HYDE_INIT, 1)
    results.append("Change 2a OK: HyDESearchTool.__init__ reranker removed")
else:
    results.append("Change 2a FAIL: HyDE __init__ not found")

# Replace the cross-encoder block + old output block with distance threshold
OLD_RERANK_BLOCK = (
    "        pairs = [[query, c.get(\"text\", \"\")[:400]] for c in candidates]\n"
    "        scores = self._reranker.predict(pairs, show_progress_bar=False)\n"
    "        ranked = sorted(\n"
    "            zip(scores, candidates), key=lambda x: x[0], reverse=True\n"
    "        )\n"
    "        # Only keep results where cross-encoder is reasonably confident\n"
    "        # Cross-encoder scores are logits; threshold ~0 means plausibly relevant\n"
    "        SCORE_THRESHOLD = -1.0\n"
    "        top = [\n"
    "            c for score, c in ranked[:self.top_k_final]\n"
    "            if score > SCORE_THRESHOLD\n"
    "        ]\n"
    "        if not top:\n"
    "            # Fallback: take top-2 regardless of threshold\n"
    "            top = [c for _, c in ranked[:2]]\n"
    "\n"
    "        self._last_citations = [c.get(\"citation\", \"\") for c in top]\n"
    "        parts = []\n"
    "        for c in top:\n"
    "            cit  = c.get(\"citation\", \"?\")\n"
    "            text = c.get(\"text\", \"\")[:self.max_excerpt]\n"
    "            parts.append(f\"[CITATION_KEY: {cit}]\\n{text}\")\n"
    "\n"
    "        header = (\n"
    "            f\"Dense search found {len(top)} results. \"\n"
    "            f\"Hypothetical used: \\\"{hypothetical[:80]}...\\\"\\n\\n\"\n"
    "        )\n"
    "        return header + \"\\n\\n\".join(parts)\n"
)

NEW_DIST_BLOCK = (
    "        # Step 4: filter by cosine distance threshold\n"
    "        # LanceDB returns _distance: lower = more similar (cosine)\n"
    "        DIST_THRESHOLD = 0.50\n"
    "        top = [c for c in candidates if c.get(\"_distance\", 1.0) < DIST_THRESHOLD]\n"
    "        if not top:\n"
    "            top = candidates[:3]  # always return at least 3\n"
    "\n"
    "        self._last_citations = [c.get(\"citation\", \"\") for c in top]\n"
    "        parts = []\n"
    "        for c in top:\n"
    "            cit  = c.get(\"citation\", \"?\")\n"
    "            text = c.get(\"text\", \"\")[:self.max_excerpt]\n"
    "            dist = c.get(\"_distance\", 1.0)\n"
    "            parts.append(\n"
    "                f\"[CITATION_KEY: {cit}]  (relevance: {1-dist:.2f})\\n{text}\"\n"
    "            )\n"
    "        return (\n"
    "            f\"Found {len(top)} relevant court sections.\\n\\n\"\n"
    "            + \"\\n\\n\".join(parts)\n"
    "        )\n"
)

if OLD_RERANK_BLOCK in s:
    s = s.replace(OLD_RERANK_BLOCK, NEW_DIST_BLOCK, 1)
    results.append("Change 2b OK: cross-encoder replaced with distance threshold")
else:
    results.append("Change 2b FAIL: rerank block not found")

set_src(nb["cells"][tools_idx], s)

# ══════════════════════════════════════════════════════════════════════════
# Change 3: HyDE instantiation — remove reranker, top_k_final
# ══════════════════════════════════════════════════════════════════════════
inst_idx = 16
s = src(nb["cells"][inst_idx])

OLD_INST = (
    "hyde_tool = HyDESearchTool(\n"
    "    llm         = llm,\n"
    "    embed_model = _embed_model,\n"
    "    lance_table = _lance_table,\n"
    "    reranker    = _reranker,\n"
    "    top_k_dense = CONFIG[\"top_k_hyde\"],\n"
    "    nprobes     = CONFIG[\"nprobes\"],\n"
    "    top_k_final = CONFIG[\"top_k_rerank\"],\n"
    "    hyde_tokens = CONFIG[\"hyde_max_tokens\"],\n"
    "    max_excerpt = CONFIG[\"max_observation_chars\"],\n"
    ")"
)

NEW_INST = (
    "hyde_tool = HyDESearchTool(\n"
    "    llm         = llm,\n"
    "    embed_model = _embed_model,\n"
    "    lance_table = _lance_table,\n"
    "    top_k_dense = CONFIG[\"top_k_hyde\"],\n"
    "    nprobes     = CONFIG[\"nprobes\"],\n"
    "    hyde_tokens = CONFIG[\"hyde_max_tokens\"],\n"
    "    max_excerpt = CONFIG[\"max_observation_chars\"],\n"
    ")"
)

if OLD_INST in s:
    s = s.replace(OLD_INST, NEW_INST, 1)
    set_src(nb["cells"][inst_idx], s)
    results.append("Change 3 OK: HyDE instantiation updated (reranker removed)")
else:
    results.append("Change 3 FAIL: HyDE instantiation not found")

# ══════════════════════════════════════════════════════════════════════════
# Change 4: CitationLookupTool — add _fr_to_de + _normalize method
# ══════════════════════════════════════════════════════════════════════════
s = src(nb["cells"][tools_idx])

# Add _fr_to_de to CitationLookupTool.__init__
OLD_LOOKUP_INIT_END = (
    "        self.max_prefix_results = max_prefix_results\n"
    "        self._last_citations: list[str] = []\n"
)
NEW_LOOKUP_INIT_END = (
    "        self.max_prefix_results = max_prefix_results\n"
    "        self._last_citations: list[str] = []\n"
    "        self._fr_to_de = {\n"
    "            \"LAI\": \"IVG\", \"CO\": \"OR\", \"CC\": \"ZGB\", \"CP\": \"StGB\",\n"
    "            \"CPP\": \"StPO\", \"LPGA\": \"ATSG\", \"LTF\": \"BGG\",\n"
    "            \"LAA\": \"UVG\", \"LAVS\": \"AHVG\", \"LPP\": \"BVG\",\n"
    "        }\n"
)

if OLD_LOOKUP_INIT_END in s:
    s = s.replace(OLD_LOOKUP_INIT_END, NEW_LOOKUP_INIT_END, 1)
    results.append("Change 4a OK: _fr_to_de added to CitationLookupTool.__init__")
else:
    results.append("Change 4a FAIL: CitationLookupTool __init__ end not found")

# Add _normalize method before run(), and update run() to use it
OLD_LOOKUP_RUN_START = (
    "    def run(self, citation: str) -> str:\n"
    "        # Sanitize: take only the first line, strip trailing punctuation\n"
    "        citation = citation.strip().splitlines()[0].strip()\n"
    "        citation = citation.rstrip(':.,')  \n"
    "        # Strip \"lit. X\" or \"Ziff. X\" suffixes the agent may append\n"
    "        citation = re.sub(r'\\s+lit\\.\\s+\\w+.*$', '', citation).strip()\n"
    "        citation = re.sub(r'\\s+Ziff\\.\\s+\\w+.*$', '', citation).strip()\n"
    "        if not citation:\n"
    "            return \"Error: empty citation string.\"\n"
    "\n"
    "        self._last_citations = []\n"
)

# Build the exact string from what we saw in the file
OLD_LOOKUP_RUN_START2 = (
    "    def run(self, citation: str) -> str:\n"
    "        # Sanitize: take only the first line, strip trailing punctuation\n"
    "        citation = citation.strip().splitlines()[0].strip()\n"
    "        citation = citation.rstrip(':.,')  \n"
)

# Try to find the exact inline sanitize block as seen in the file
idx = s.find("    def run(self, citation: str) -> str:\n        # Sanitize: take only the first line")
if idx != -1:
    # Find end of sanitize block (up to self._last_citations = [])
    end_marker = "        self._last_citations = []\n"
    end_idx = s.find(end_marker, idx)
    if end_idx != -1:
        old_block = s[idx:end_idx + len(end_marker)]
        new_block = (
            "    def _normalize(self, citation: str) -> str:\n"
            "        import re as _re\n"
            "        for stop in ['\\n', 'Thought:', 'Action:', 'Now ', 'Found ',\n"
            "                     'Looking', 'Now look', 'Next']:\n"
            "            if stop in citation:\n"
            "                citation = citation[:citation.index(stop)]\n"
            "        citation = citation.strip()\n"
            "        citation = _re.sub(r'\\s*\\([^)]*\\)', '', citation).strip()\n"
            "        citation = _re.sub(r'[\\s]*[:\\.\\,;]$', '', citation).strip()\n"
            "        citation = _re.sub(r'\\s+lit\\.\\s+\\w+.*$', '', citation).strip()\n"
            "        citation = _re.sub(r'\\s+Ziff\\.\\s+\\w+.*$', '', citation).strip()\n"
            "        tokens = citation.split()\n"
            "        if tokens and tokens[-1] in self._fr_to_de:\n"
            "            tokens[-1] = self._fr_to_de[tokens[-1]]\n"
            "            citation = ' '.join(tokens)\n"
            "        return citation\n"
            "\n"
            "    def run(self, citation: str) -> str:\n"
            "        citation = self._normalize(citation)\n"
            "        if not citation:\n"
            "            return \"Error: empty citation string.\"\n"
            "\n"
            "        self._last_citations = []\n"
        )
        s = s[:idx] + new_block + s[end_idx + len(end_marker):]
        results.append("Change 4b OK: _normalize method added, run() updated")
    else:
        results.append("Change 4b FAIL: end_marker not found")
else:
    results.append("Change 4b FAIL: CitationLookupTool.run() start not found")

set_src(nb["cells"][tools_idx], s)

# ══════════════════════════════════════════════════════════════════════════
# Change 5: Agent prompt — replace STEP 3 + 3b, update RULES
# ══════════════════════════════════════════════════════════════════════════
agent_idx = 18
s = src(nb["cells"][agent_idx])

OLD_STEP3_BLOCK = (
    "STEP 3 - SEARCH FOR CASE LAW\n"
    "After looking up at least 5 articles, call search_courts with German keywords.\n"
    "Example: search_courts(\"Untersuchungshaft Kollusionsgefahr Verhaeltnismaessigkeit\")\n"
    "\n"
    "STEP 3b - DENSE SEMANTIC COURT SEARCH\n"
    "After BM25 search, call hyde_search_courts with German legal concepts.\n"
    "\n"
    "Action: hyde_search_courts\n"
    "Action Input: Kollusionsgefahr konkrete Gefaehrdung Zeugen Untersuchungshaft\n"
    "\n"
    "The tool returns court sections ranked by semantic relevance.\n"
    "SELECT only the CITATION_KEYs whose text is clearly relevant to the query.\n"
    "Do NOT blindly include all results - read each excerpt and judge relevance.\n"
    "A result is relevant if its text directly addresses the legal question asked.\n"
    "Include only citations where you can confirm relevance from the excerpt.\n"
    "\n"
)

NEW_STEP3_BLOCK = (
    "STEP 3 - MANDATORY COURT SEARCHES (do both, always)\n"
    "\n"
    "First, BM25 keyword search in German:\n"
    "  Action: search_courts\n"
    "  Action Input: Untersuchungshaft Kollusionsgefahr Verhaeltnismaessigkeit Verlaengerung\n"
    "\n"
    "Then, dense semantic search in German (MANDATORY - always call this):\n"
    "  Action: hyde_search_courts\n"
    "  Action Input: Kollusionsgefahr konkrete Gefaehrdung Zeugen Untersuchungshaft\n"
    "\n"
    "SEARCH RULES - violations will produce wrong answers:\n"
    "- Both queries MUST be in German legal terminology only\n"
    "- WRONG: \"Pretrial detention Recht auf Anhoerung\"  (mixed languages)\n"
    "- RIGHT:  \"Untersuchungshaft Recht auf Anhoerung ausreichender Tatverdacht\"\n"
    "- hyde_search_courts query must NOT contain citation strings\n"
    "- After hyde_search_courts returns, call lookup_citation for each\n"
    "  BGE shown that seems relevant to the legal question\n"
    "\n"
)

if OLD_STEP3_BLOCK in s:
    s = s.replace(OLD_STEP3_BLOCK, NEW_STEP3_BLOCK, 1)
    results.append("Change 5a OK: STEP 3 + 3b replaced with mandatory dual-search")
else:
    results.append("Change 5a FAIL: STEP 3 block not found")

# Update RULES: "at least 8" → new 3-line rules
OLD_RULES = (
    "RULES:\n"
    "- You MUST call at least 8 lookup_citation calls before concluding\n"
    "- Never output a citation you did not retrieve text for via a tool call\n"
    "- If lookup returns \"not found\", try shorter prefix: \"Art. 221 StPO\" finds all Abs.\n"
    "- Search courts ONLY in German, with 3-5 keywords\n"
)
NEW_RULES = (
    "RULES:\n"
    "- You MUST call at least 6 lookup_citation calls before concluding\n"
    "- You MUST call search_courts exactly once (German query)\n"
    "- You MUST call hyde_search_courts exactly once (German query)\n"
    "- Never output a citation you did not retrieve text for via a tool call\n"
    "- If lookup returns \"not found\", try shorter prefix: \"Art. 221 StPO\" finds all Abs.\n"
    "- All search queries MUST be in German legal terminology only\n"
)

if OLD_RULES in s:
    s = s.replace(OLD_RULES, NEW_RULES, 1)
    results.append("Change 5b OK: RULES updated (6 lookups + mandatory searches)")
else:
    results.append("Change 5b FAIL: RULES block not found")

set_src(nb["cells"][agent_idx], s)

# Change 6 already done in Change 1 (max_iterations 8→6)
results.append("Change 6 OK: max_iterations=6 (applied in Change 1)")

# ══════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n".join(results))
print(f"\nSaved: {NB_PATH}")

# Verify
nb2 = json.load(open(NB_PATH, encoding="utf-8"))
def src2(c): s=c["source"]; return "".join(s) if isinstance(s,list) else s
all_src = "".join(src2(c) for c in nb2["cells"])
print("\nVerification:")
checks = [
    ("top_k_hyde=20",             '"top_k_hyde": 20,' in all_src),
    ("hyde_max_tokens=120",       '"hyde_max_tokens": 120,' in all_src),
    ("max_iterations=6",          '"max_iterations": 6,' in all_src),
    ("no reranker in HyDE init",  "reranker," not in src2(nb2["cells"][11])[src2(nb2["cells"][11]).find("class HyDESearchTool"):]),
    ("distance threshold",        "DIST_THRESHOLD = 0.50" in all_src),
    ("no reranker in instantiation", "reranker    = _reranker" not in src2(nb2["cells"][16])),
    ("_normalize method",         "def _normalize" in all_src),
    ("_fr_to_de dict",            "_fr_to_de" in all_src),
    ("mandatory dual-search step","MANDATORY COURT SEARCHES" in all_src),
    ("6 lookup rule",             "at least 6 lookup_citation" in all_src),
    ("hyde mandatory rule",       "MUST call hyde_search_courts" in all_src),
]
for label, ok in checks:
    print(f"  {'OK' if ok else 'FAIL'} {label}")
