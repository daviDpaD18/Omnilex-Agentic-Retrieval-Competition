import json

NB_PATH = "c:/Users/david/Desktop/project_ssl/Omnilex-Agentic-Retrieval-Competition/notebooks/02_agentic_retrieval_baseline.ipynb"

nb = json.load(open(NB_PATH, encoding="utf-8"))

# ── Step 0: remove old lookup cell (cell 10) ─────────────────────────────
old_lookup_idx = next(
    i for i, c in enumerate(nb["cells"])
    if "Build O(1) citation lookup dicts" in "".join(c["source"])
       or "LookupCitationTool" in "".join(c["source"])
)
nb["cells"].pop(old_lookup_idx)
print(f"Removed old lookup cell (was at index {old_lookup_idx})")

# ── Step 1: insert lookup dict cell after courts_index loading cell ───────
courts_cell_idx = next(
    i for i, c in enumerate(nb["cells"])
    if "courts_index = get_or_build_index" in "".join(c["source"])
)

LOOKUP_DICT_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── Citation lookup dicts — O(1) direct access by citation key ────────────\n",
        "# RAM cost: laws ~80MB, courts ~800MB — well within 16GB budget\n",
        "print('Building citation lookup dicts...')\n",
        "\n",
        "laws_lookup = {\n",
        "    doc['citation']: doc['text']\n",
        "    for doc in laws_index.documents\n",
        "    if doc.get('citation')\n",
        "}\n",
        "\n",
        "courts_lookup = {\n",
        "    doc['citation']: doc['text']\n",
        "    for doc in courts_index.documents\n",
        "    if doc.get('citation')\n",
        "}\n",
        "\n",
        "print(f'  laws_lookup   : {len(laws_lookup):,} entries')\n",
        "print(f'  courts_lookup : {len(courts_lookup):,} entries')\n",
        "\n",
        "# Verify key gold citations are reachable\n",
        "test_keys = ['Art. 221 Abs. 1 StPO', 'Art. 100 Abs. 1 BGG', 'Art. 37 Abs. 1 StBOG']\n",
        "for k in test_keys:\n",
        "    found = k in laws_lookup\n",
        "    print(f\"  {'OK' if found else 'MISSING'} laws_lookup['{k}']\")\n",
    ]
}

nb["cells"].insert(courts_cell_idx + 1, LOOKUP_DICT_CELL)
print(f"Inserted lookup dict cell at index {courts_cell_idx + 1}")

# ── Step 2+3: update tools cell — append CitationLookupTool + update TOOLS ─
tools_cell_idx = next(
    i for i, c in enumerate(nb["cells"])
    if "class LawSearchTool" in "".join(c["source"])
)

CITATION_TOOL_CLASS = '''

class CitationLookupTool:
    """Direct O(1) lookup of a specific citation by its exact key.

    Use this when you already know or strongly suspect a specific citation
    is relevant — e.g. Art. 221 Abs. 1 StPO for a pre-trial detention case.
    Faster and more precise than BM25 search for known citations.
    """

    name: str = "lookup_citation"
    description: str = """Look up the exact text of a specific legal citation by its key.
Input: A citation string, e.g. "Art. 221 Abs. 1 StPO" or "BGE 137 IV 122 E. 6.2"
Output: The full text of that provision, or prefix matches if exact key not found.

Use this tool when you know which article or decision section is likely relevant.
For discovering unknown relevant law, use search_laws or search_courts instead.
Example inputs: "Art. 221 Abs. 1 StPO", "Art. 100 Abs. 1 BGG", "1B_90/2021 E. 2.1"
"""

    def __init__(
        self,
        laws_lut: dict,
        courts_lut: dict,
        max_excerpt_length: int = 500,
        max_prefix_results: int = 8,
    ):
        self._laws = laws_lut
        self._courts = courts_lut
        self.max_excerpt_length = max_excerpt_length
        self.max_prefix_results = max_prefix_results
        self._last_citations: list[str] = []

    def __call__(self, citation: str) -> str:
        return self.run(citation)

    def run(self, citation: str) -> str:
        citation = citation.strip()
        if not citation:
            return "Error: empty citation string."

        self._last_citations = []

        # 1. Exact match — check both stores
        for store_name, store in [("laws", self._laws), ("courts", self._courts)]:
            if citation in store:
                text = store[citation]
                if len(text) > self.max_excerpt_length:
                    text = text[:self.max_excerpt_length] + "..."
                self._last_citations = [citation]
                return f"[{citation}]\\n{text}"

        # 2. Prefix match — e.g. "Art. 221 StPO" finds "Art. 221 Abs. 1 StPO" etc.
        # Also handles "BGE 137 IV 122" finding all E. sections
        candidates = []
        for store in [self._laws, self._courts]:
            candidates += [
                (k, v) for k, v in store.items()
                if k.startswith(citation)
            ]

        if candidates:
            self._last_citations = [k for k, _ in candidates[:self.max_prefix_results]]
            parts = []
            for k, v in candidates[:self.max_prefix_results]:
                if len(v) > self.max_excerpt_length:
                    v = v[:self.max_excerpt_length] + "..."
                parts.append(f"[{k}]\\n{v}")
            return (
                f"Exact citation '{citation}' not found. "
                f"Prefix matches ({len(candidates)} total, showing "
                f"{min(len(candidates), self.max_prefix_results)}):\\n\\n"
                + "\\n\\n".join(parts)
            )

        return (
            f"Citation '{citation}' not found. "
            f"Check spelling — example valid formats: "
            f"'Art. 221 Abs. 1 StPO', 'BGE 137 IV 122 E. 6.2', '1B_90/2021 E. 2.1'"
        )

    def get_last_citations(self) -> list[str]:
        return list(self._last_citations)
'''

OLD_INSTANTIATION = '''# Create tools
law_tool = LawSearchTool(
    index=laws_index,
    top_k=CONFIG["top_k_laws"],
    max_excerpt_length=300,
)

court_tool = CourtSearchTool(
    index=courts_index,
    top_k=CONFIG["top_k_courts"],
    max_excerpt_length=300,
)

# Tool registry
TOOLS = {
    "search_laws": law_tool,
    "search_courts": court_tool,
}

print("Tools registered:")
for name, tool in TOOLS.items():
    print(f"  - {name}: {tool.description.split(chr(10))[0]}")'''

NEW_INSTANTIATION = '''# Create tools
law_tool = LawSearchTool(
    index=laws_index,
    top_k=CONFIG["top_k_laws"],
    max_excerpt_length=300,
)

court_tool = CourtSearchTool(
    index=courts_index,
    top_k=CONFIG["top_k_courts"],
    max_excerpt_length=300,
)

lookup_tool = CitationLookupTool(
    laws_lut=laws_lookup,
    courts_lut=courts_lookup,
    max_excerpt_length=CONFIG.get("max_observation_chars", 1200),
    max_prefix_results=8,
)

# Tool registry
TOOLS = {
    "search_laws": law_tool,
    "search_courts": court_tool,
    "lookup_citation": lookup_tool,
}

print("Tools registered:")
for name, tool in TOOLS.items():
    print(f"  - {name}: {tool.description.split(chr(10))[0]}")'''

src = "".join(nb["cells"][tools_cell_idx]["source"])
assert "class CourtSearchTool" in src, "CourtSearchTool not found in tools cell"
assert OLD_INSTANTIATION in src, "Could not find old tool instantiation block"

# Append CitationLookupTool class before the instantiation block
src = src.replace(OLD_INSTANTIATION, CITATION_TOOL_CLASS + "\n\n" + NEW_INSTANTIATION)
nb["cells"][tools_cell_idx]["source"] = src
print(f"Updated tools cell at index {tools_cell_idx}: added CitationLookupTool + updated TOOLS dict")

# ── Step 4: fix agent system prompt ──────────────────────────────────────
agent_cell_idx = next(
    i for i, c in enumerate(nb["cells"])
    if "AGENT_SYSTEM_PROMPT" in "".join(c["source"])
)

NEW_PROMPT_TOOLS = '''Du bist ein Schweizer Rechtsrecherche-Assistent mit Zugang zu drei Tools:

1. search_laws(query): Durchsuche Schweizer Bundesgesetze (SR/Systematische Rechtssammlung)
   - Gibt relevante Gesetzesbestimmungen mit Zitaten und Textauszügen zurück
   - Verwende für konzeptuelle Suche: Kodizes, Gesetze, Verordnungen

2. search_courts(query): Durchsuche Schweizer Bundesgerichtsentscheide (BGE)
   - Gibt relevante Rechtsprechung mit Zitaten und Auszügen zurück
   - Verwende für Gerichtsentscheide und Präzedenzfälle

3. lookup_citation(citation): Direkter O(1)-Zugriff auf eine bekannte Zitierungsangabe
   - Verwende dies, wenn du eine spezifische Zitation bereits kennst oder stark vermutest
   - Unterstützt exakte Schlüssel UND Präfix-Suche
   - Beispiele: "Art. 221 Abs. 1 StPO", "BGE 137 IV 122 E. 6.2", "1B_90/2021"
   - Präfix-Suche: "Art. 221 StPO" findet alle Absätze; "BGE 137 IV 122" findet alle E.-Abschnitte'''

src2 = "".join(nb["cells"][agent_cell_idx]["source"])

# Replace everything from the opening triple-quote to WICHTIG
import re
src2 = re.sub(
    r'(AGENT_SYSTEM_PROMPT = """)(.*?)(WICHTIG:)',
    lambda m: m.group(1) + NEW_PROMPT_TOOLS + '\n\n' + m.group(3),
    src2,
    flags=re.DOTALL,
    count=1,
)
nb["cells"][agent_cell_idx]["source"] = src2
print(f"Updated agent system prompt at cell index {agent_cell_idx}")

# ── Step 5: insert test cell after tools cell ─────────────────────────────
TEST_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── Step 5: Verify CitationLookupTool before running eval ─────────────────\n",
        "test_cases = [\n",
        "    'Art. 221 Abs. 1 StPO',   # exact law\n",
        "    'Art. 100 Abs. 1 BGG',    # exact law\n",
        "    'Art. 37 Abs. 1 StBOG',   # exact law\n",
        "    'BGE 137 IV 122',         # prefix — all E. sections of that decision\n",
        "    '1B_90/2021',             # prefix — docket\n",
        "]\n",
        "print(f'  {\"Citation\":<40} {\"Status\":<8} Preview')\n",
        "print('  ' + '-' * 80)\n",
        "for tc in test_cases:\n",
        "    result = lookup_tool(tc)\n",
        "    status = 'OK' if 'not found' not in result.lower() else 'FAIL'\n",
        "    preview = result[:80].replace('\\n', ' ')\n",
        "    print(f'  {tc!r:<40} [{status}]   {preview}')\n",
    ]
}

# Re-find tools cell index (may have shifted after insertions)
tools_cell_idx = next(
    i for i, c in enumerate(nb["cells"])
    if "class CitationLookupTool" in "".join(c["source"])
)
nb["cells"].insert(tools_cell_idx + 1, TEST_CELL)
print(f"Inserted test cell at index {tools_cell_idx + 1}")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

total = len(nb["cells"])
print(f"\nDone. Notebook now has {total} cells.")
