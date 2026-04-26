import json, re

NB_PATH = "c:/Users/david/Desktop/project_ssl/Omnilex-Agentic-Retrieval-Competition/notebooks/02_agentic_retrieval_baseline.ipynb"

# ── New cell: lookup tool definition + test ───────────────────────────────
TOOL_SOURCE = [
    "# ── Build O(1) citation lookup dicts from loaded index documents ─────────\n",
    "print('Building citation lookup dicts...')\n",
    "laws_lookup   = {doc['citation']: doc['text'] for doc in laws_index.documents}\n",
    "courts_lookup = {doc['citation']: doc['text'] for doc in courts_index.documents}\n",
    "print(f'  laws_lookup   : {len(laws_lookup):,} entries')\n",
    "print(f'  courts_lookup : {len(courts_lookup):,} entries')\n",
    "\n",
    "\n",
    "class LookupCitationTool:\n",
    "    \"\"\"O(1) exact-key lookup for legal citations, with prefix fallback.\"\"\"\n",
    "\n",
    "    name: str = 'lookup_citation'\n",
    "    description: str = (\n",
    "        'Retrieve the exact text of a specific legal citation by its key.\\n'\n",
    "        'Use this when you already know or strongly suspect a specific citation.\\n'\n",
    "        'Falls back to prefix matching if the exact key is not found.\\n'\n",
    "        'Example inputs: \"Art. 221 Abs. 1 StPO\", \"BGE 137 IV 122 E. 6.2\", '\n",
    "        '\"1B_90/2021 E. 2.1\", \"Art. 221 StPO\" (prefix)'\n",
    "    )\n",
    "\n",
    "    def __init__(\n",
    "        self,\n",
    "        laws_lut: dict,\n",
    "        courts_lut: dict,\n",
    "        max_prefix_results: int = 5,\n",
    "        max_text_chars: int = 600,\n",
    "    ):\n",
    "        self._laws   = laws_lut\n",
    "        self._courts = courts_lut\n",
    "        self._max_prefix = max_prefix_results\n",
    "        self._max_text   = max_text_chars\n",
    "        self._last_citations: list[str] = []\n",
    "\n",
    "    def __call__(self, citation: str) -> str:\n",
    "        return self.run(citation)\n",
    "\n",
    "    def run(self, citation: str) -> str:\n",
    "        if not citation or not citation.strip():\n",
    "            self._last_citations = []\n",
    "            return 'Error: empty citation string.'\n",
    "\n",
    "        key = citation.strip()\n",
    "\n",
    "        # ── Exact lookup ──────────────────────────────────────────────────\n",
    "        text = self._laws.get(key) or self._courts.get(key)\n",
    "        if text:\n",
    "            self._last_citations = [key]\n",
    "            snippet = text[:self._max_text]\n",
    "            if len(text) > self._max_text:\n",
    "                snippet += '...'\n",
    "            return f'[{key}]\\n{snippet}'\n",
    "\n",
    "        # ── Prefix fallback ───────────────────────────────────────────────\n",
    "        candidates: list[tuple[str, str]] = []\n",
    "        for store in (self._laws, self._courts):\n",
    "            candidates += [\n",
    "                (k, v) for k, v in store.items()\n",
    "                if k.startswith(key)\n",
    "            ]\n",
    "            if len(candidates) >= self._max_prefix:\n",
    "                break\n",
    "        candidates = candidates[:self._max_prefix]\n",
    "\n",
    "        if candidates:\n",
    "            self._last_citations = [k for k, _ in candidates]\n",
    "            parts = []\n",
    "            for k, v in candidates:\n",
    "                snippet = v[:self._max_text]\n",
    "                if len(v) > self._max_text:\n",
    "                    snippet += '...'\n",
    "                parts.append(f'[{k}]\\n{snippet}')\n",
    "            header = f\"Exact key not found. Prefix matches for '{key}':\"\n",
    "            return header + '\\n\\n' + '\\n\\n'.join(parts)\n",
    "\n",
    "        self._last_citations = []\n",
    "        return f\"Citation '{key}' not found in index (no exact or prefix match).\"\n",
    "\n",
    "    def get_last_citations(self) -> list[str]:\n",
    "        return list(self._last_citations)\n",
    "\n",
    "\n",
    "# ── Instantiate and register ──────────────────────────────────────────────\n",
    "lookup_tool = LookupCitationTool(\n",
    "    laws_lut=laws_lookup,\n",
    "    courts_lut=courts_lookup,\n",
    "    max_prefix_results=5,\n",
    "    max_text_chars=600,\n",
    ")\n",
    "TOOLS['lookup_citation'] = lookup_tool\n",
    "print('\\nTools registered:', list(TOOLS.keys()))\n",
    "\n",
    "\n",
    "# ── Manual tests ─────────────────────────────────────────────────────────\n",
    "TEST_CALLS = [\n",
    "    # (label, citation_string)\n",
    "    ('Exact — law',    'Art. 221 Abs. 1 StPO'),\n",
    "    ('Exact — law',    'Art. 100 Abs. 1 BGG'),\n",
    "    ('Exact — court',  'BGE 137 IV 122 E. 6.2'),\n",
    "    ('Prefix — law',   'Art. 221 StPO'),\n",
    "    ('Prefix — law',   'Art. 100 BGG'),\n",
    "    ('Prefix — court', 'BGE 137 IV 122'),\n",
    "]\n",
    "\n",
    "print('\\n' + '=' * 70)\n",
    "print('  lookup_citation — manual test results')\n",
    "print('=' * 70)\n",
    "\n",
    "for label, cit in TEST_CALLS:\n",
    "    print(f'\\n  [{label}] lookup_citation(\"{cit}\")')\n",
    "    print('  ' + '-' * 66)\n",
    "    result = lookup_tool(cit)\n",
    "    for line in result.splitlines():\n",
    "        print(f'  {line}')\n",
    "    found_cits = lookup_tool.get_last_citations()\n",
    "    print(f'  -> Citations captured: {found_cits}')\n",
    "\n",
    "print('\\n' + '=' * 70)\n",
    "print('  Test complete')\n",
    "print('=' * 70)\n",
]

# ── Updated agent system prompt snippet (3-tool version) ──────────────────
PROMPT_ADDITION = (
    "\\n4. lookup_citation(citation): Direkte Textabfrage für eine bekannte Zitierungsangabe.\\n"
    "   - Verwende dies, wenn du eine spezifische Zitation bereits kennst oder stark vermutest.\\n"
    "   - Unterstützt exakte Schlüssel und Präfix-Suche (z.B. 'Art. 221 StPO')."
)

nb = json.load(open(NB_PATH, encoding="utf-8"))

# ── Find cell 9 (LawSearchTool / CourtSearchTool / TOOLS dict) ────────────
tool_cell_idx = next(
    i for i, c in enumerate(nb["cells"])
    if "LawSearchTool" in "".join(c["source"])
)
print(f"Found tools cell at index {tool_cell_idx}")

# ── Insert new tool cell right after the existing tools cell ──────────────
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": TOOL_SOURCE,
}
nb["cells"].insert(tool_cell_idx + 1, new_cell)
print(f"Inserted lookup tool cell at index {tool_cell_idx + 1}")

# ── Find and patch the AGENT_SYSTEM_PROMPT cell ───────────────────────────
# After insertion, the agent cell shifted by 1
agent_cell_idx = next(
    i for i, c in enumerate(nb["cells"])
    if "AGENT_SYSTEM_PROMPT" in "".join(c["source"])
)
print(f"Found agent prompt cell at index {agent_cell_idx}")

src = nb["cells"][agent_cell_idx]["source"]
src_str = "".join(src) if isinstance(src, list) else src

# Find the tool list description block and append the 4th tool entry
OLD_SNIPPET = "2. search_courts(query): Durchsuche Schweizer Bundesgerichtsentscheide (BGE)"
NEW_SNIPPET = (
    "2. search_courts(query): Durchsuche Schweizer Bundesgerichtsentscheide (BGE)\n"
    "\n"
    "3. lookup_citation(citation): Direkte Textabfrage für eine bekannte Zitierungsangabe.\n"
    "   - Verwende dies, wenn du eine spezifische Zitation bereits kennst oder stark vermutest.\n"
    "   - Unterstützt exakte Schlüssel und Präfix-Suche (z.B. \"Art. 221 StPO\", \"BGE 137 IV 122\")"
)

if OLD_SNIPPET in src_str:
    patched = src_str.replace(OLD_SNIPPET, NEW_SNIPPET, 1)
    # Also update the tool count reference
    patched = patched.replace(
        "Du bist ein Schweizer Rechtsrecherche-Assistent mit Zugang zu zwei Such-Tools:",
        "Du bist ein Schweizer Rechtsrecherche-Assistent mit Zugang zu drei Such-Tools:"
    )
    nb["cells"][agent_cell_idx]["source"] = patched
    print(f"Patched AGENT_SYSTEM_PROMPT at cell {agent_cell_idx}")
else:
    print("WARNING: Could not find expected snippet in AGENT_SYSTEM_PROMPT — prompt not patched")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

total = len(nb["cells"])
print(f"\nDone. Notebook now has {total} cells.")
print(f"Lookup tool cell at index {tool_cell_idx + 1}")
