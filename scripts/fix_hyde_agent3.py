"""
Three targeted fixes to 04_hyde_agent.ipynb:
  Fix 1: CitationLookupTool prefix branch -> return keys only (no text),
          max_prefix_results=3, exact match capped at 200 chars
  Fix 2: Replace AGENT_SYSTEM_PROMPT with new 5-step procedure
          (hyde_search_courts fires as mandatory STEP 2)
  Fix 3: Domain hints for IVG/ATSG and StPO cases embedded in prompt
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

NB_PATH = "notebooks/04_hyde_agent.ipynb"
nb = json.load(open(NB_PATH, encoding="utf-8"))

def src(c): s = c["source"]; return "".join(s) if isinstance(s, list) else s
def set_src(c, s): c["source"] = s

results = []

# ── Fix 1: CitationLookupTool — keys-only prefix, exact cap 200 chars ────────
tools_idx = 11
s = src(nb["cells"][tools_idx])

# 1a. Exact match: cap at 200 chars instead of max_excerpt_length
OLD_EXACT = (
    "        # 1. Exact match — check both stores\n"
    "        for store_name, store in [(\"laws\", self._laws), (\"courts\", self._courts)]:\n"
    "            if citation in store:\n"
    "                text = store[citation]\n"
    "                if len(text) > self.max_excerpt_length:\n"
    "                    text = text[:self.max_excerpt_length] + \"...\"\n"
    "                self._last_citations = [citation]\n"
    "                return f\"[{citation}]\\n{text}\"\n"
)
NEW_EXACT = (
    "        # 1. Exact match — check both stores, cap at 200 chars\n"
    "        for store_name, store in [(\"laws\", self._laws), (\"courts\", self._courts)]:\n"
    "            if citation in store:\n"
    "                text = store[citation]\n"
    "                if len(text) > 200:\n"
    "                    text = text[:200] + \"...\"\n"
    "                self._last_citations = [citation]\n"
    "                return f\"[{citation}]\\n{text}\"\n"
)
if OLD_EXACT in s:
    s = s.replace(OLD_EXACT, NEW_EXACT, 1)
    results.append("Fix 1a OK: exact match capped at 200 chars")
else:
    results.append("Fix 1a FAIL: exact match block not found")

# 1b. Prefix branch: return keys only, no text
OLD_PREFIX_RETURN = (
    "        if candidates:\n"
    "            self._last_citations = [k for k, _ in candidates[:self.max_prefix_results]]\n"
    "            parts = []\n"
    "            for k, v in candidates[:self.max_prefix_results]:\n"
    "                if len(v) > self.max_excerpt_length:\n"
    "                    v = v[:self.max_excerpt_length] + \"...\"\n"
    "                parts.append(f\"[{k}]\\n{v}\")\n"
    "            return (\n"
    "                f\"Exact citation '{citation}' not found. \"\n"
    "                f\"Prefix matches ({len(candidates)} total, showing \"\n"
    "                f\"{min(len(candidates), self.max_prefix_results)}):\\n\\n\"\n"
    "                + \"\\n\\n\".join(parts)\n"
    "            )\n"
)
NEW_PREFIX_RETURN = (
    "        if candidates:\n"
    "            self._last_citations = [k for k, _ in candidates[:self.max_prefix_results]]\n"
    "            key_list = \"\\n\".join(\n"
    "                f\"  CITATION_KEY: {k}\"\n"
    "                for k, _ in candidates[:self.max_prefix_results]\n"
    "            )\n"
    "            total = len(candidates)\n"
    "            return (\n"
    "                f\"Prefix '{citation}' matched {total} citations. \"\n"
    "                f\"Top {min(total, self.max_prefix_results)} keys:\\n\"\n"
    "                f\"{key_list}\\n\\n\"\n"
    "                f\"Call lookup_citation with the EXACT key you need, \"\n"
    "                f'e.g. lookup_citation(\"Art. 100 Abs. 1 BGG\")'\n"
    "            )\n"
)
if OLD_PREFIX_RETURN in s:
    s = s.replace(OLD_PREFIX_RETURN, NEW_PREFIX_RETURN, 1)
    results.append("Fix 1b OK: prefix branch returns keys only")
else:
    results.append("Fix 1b FAIL: prefix return block not found")

# 1c. Change max_prefix_results=8 to max_prefix_results=3 in instantiation
OLD_MAX_PREFIX = "    max_prefix_results=8,\n)"
NEW_MAX_PREFIX = "    max_prefix_results=3,\n)"
if OLD_MAX_PREFIX in s:
    s = s.replace(OLD_MAX_PREFIX, NEW_MAX_PREFIX, 1)
    results.append("Fix 1c OK: max_prefix_results 8 -> 3")
else:
    results.append("Fix 1c FAIL: max_prefix_results=8 not found in instantiation")

set_src(nb["cells"][tools_idx], s)

# ── Fix 2 + 3: New AGENT_SYSTEM_PROMPT with 5-step + domain hints ────────────
prompt_idx = 18
s = src(nb["cells"][prompt_idx])

# Find the AGENT_SYSTEM_PROMPT string start and end
start_marker = 'AGENT_SYSTEM_PROMPT = """'
end_marker = '"""\n\n\ndef parse_all_agent_actions'
alt_end_marker = '"""\n\n\ndef parse_all_agent_actions'

if start_marker not in s:
    results.append("Fix 2 FAIL: AGENT_SYSTEM_PROMPT start marker not found")
else:
    prompt_start = s.index(start_marker)
    # Find the closing triple quote
    content_start = prompt_start + len(start_marker)
    prompt_end = s.index('"""\n', content_start)

    NEW_PROMPT_BODY = (
        "You are a Swiss legal research assistant. Your ONLY job is to find citations.\n"
        "\n"
        "MANDATORY 5-STEP PROCEDURE - follow EXACTLY for every query:\n"
        "\n"
        "STEP 1 - IDENTIFY DOMAIN AND KEY ARTICLES\n"
        "Read the query. Identify the legal domain and list articles to look up.\n"
        "\n"
        "PRE-TRIAL DETENTION (Untersuchungshaft / StPO) domain:\n"
        "  Always look up: Art. 221 StPO, Art. 212 StPO, Art. 222 StPO,\n"
        "  Art. 227 StPO, Art. 393 StPO, Art. 428 StPO, Art. 100 BGG, Art. 37 StBOG\n"
        "\n"
        "INVALIDITY INSURANCE (IVG / ATSG / Invalidenversicherung) domain:\n"
        "  Always look up: Art. 8 IVG, Art. 17 IVG, Art. 28 IVG, Art. 29 IVG,\n"
        "  Art. 6 ATSG, Art. 8 ATSG, Art. 16 ATSG, Art. 61 ATSG, Art. 100 BGG\n"
        "\n"
        "STEP 2 - DENSE SEMANTIC SEARCH (MANDATORY, always second)\n"
        "Call hyde_search_courts with a German legal concept phrase (no citations).\n"
        "\n"
        "  Action: hyde_search_courts\n"
        "  Action Input: [German legal concept describing the case, ~5-10 words]\n"
        "\n"
        "RULES for hyde_search_courts:\n"
        "- Query MUST be in German legal terminology ONLY (no French, no English)\n"
        "- Query must NOT contain citation strings like 'Art. 221' or 'BGE 137'\n"
        "- Pre-trial detention example: 'Kollusionsgefahr Untersuchungshaft Verhaeltnismaessigkeit'\n"
        "- Invalidity example: 'Invaliditaetsbemessung Arbeitsfaehigkeit widerspruechliche Gutachten'\n"
        "- After it returns, call lookup_citation for each CITATION_KEY shown\n"
        "\n"
        "STEP 3 - LOOKUP EACH ARTICLE WITH EXACT KEY\n"
        "Call lookup_citation for each article identified in STEP 1.\n"
        "IMPORTANT: When a prefix lookup returns CITATION_KEY list,\n"
        "immediately call lookup_citation again with the EXACT full key.\n"
        "\n"
        "  CORRECT: lookup_citation(\"Art. 221 Abs. 1 StPO\")  <- exact key\n"
        "  CORRECT: lookup_citation(\"Art. 221 StPO\")          <- prefix, then use returned key\n"
        "  WRONG:   lookup_citation(\"Art. 221 StPO lit. b\\n\\n2. Art. 212...\")  <- multiline\n"
        "\n"
        "STEP 4 - BM25 KEYWORD SEARCH\n"
        "Call search_courts with German legal terms from the query.\n"
        "\n"
        "  Action: search_courts\n"
        "  Action Input: [German legal keywords, NOT citation strings]\n"
        "\n"
        "When search_courts returns BGE or docket citations, call lookup_citation for each.\n"
        "Do NOT use search_courts to look up docket numbers like '1B_90/2021'.\n"
        "\n"
        "STEP 5 - OUTPUT ALL CONFIRMED CITATIONS\n"
        "Format: CITATIONS: Art. 221 Abs. 1 StPO; BGE 143 IV 168 E. 5.1; ...\n"
        "\n"
        "Only emit citations whose text you actually retrieved via a tool call.\n"
        "\n"
        "RULES:\n"
        "- You MUST call hyde_search_courts exactly once (German query, no citations)\n"
        "- You MUST call at least 5 lookup_citation calls before concluding\n"
        "- You MUST call search_courts exactly once\n"
        "- Never output a citation you did not retrieve text for\n"
        "- A prefix lookup returning keys does NOT count as retrieving text\n"
        "  You must call lookup_citation again with the EXACT full key\n"
        "\n"
        "RESPONSE FORMAT:\n"
        "Thought: [reasoning]\n"
        "Action: hyde_search_courts\n"
        "Action Input: Kollusionsgefahr Untersuchungshaft Verhaeltnismaessigkeit\n"
        "\n"
        "After Observation:\n"
        "Thought: [next step]\n"
        "Action: lookup_citation\n"
        "Action Input: Art. 221 Abs. 1 StPO\n"
        "\n"
        "EXAMPLE (pre-trial detention):\n"
        "Thought: Pre-trial detention case. STEP 2 first: hyde_search_courts.\n"
        "Action: hyde_search_courts\n"
        "Action Input: Kollusionsgefahr Untersuchungshaft Verhaeltnismaessigkeit Verlaengerung\n"
        "\n"
        "Observation: Found 3 relevant court sections.\n"
        "  [CITATION_KEY: BGE 143 IV 168]...\n"
        "  [CITATION_KEY: 1B_235/2018 E. 3.1]...\n"
        "\n"
        "Thought: Good. Now STEP 3: look up each returned key and domain articles.\n"
        "Action: lookup_citation\n"
        "Action Input: BGE 143 IV 168\n"
        "\n"
        "Observation: Prefix 'BGE 143 IV 168' matched 4 citations. Top 3 keys:\n"
        "  CITATION_KEY: BGE 143 IV 168 E. 3.1\n"
        "  CITATION_KEY: BGE 143 IV 168 E. 4.2\n"
        "  CITATION_KEY: BGE 143 IV 168 E. 5.1\n"
        "Call lookup_citation with the EXACT key you need.\n"
        "\n"
        "Thought: Prefix returned keys. Must look up exact key.\n"
        "Action: lookup_citation\n"
        "Action Input: BGE 143 IV 168 E. 3.1\n"
        "\n"
        "Observation: [BGE 143 IV 168 E. 3.1] Die Untersuchungshaft ist zulaessig wenn...\n"
        "\n"
        "Thought: Now look up Art. 221 StPO domain article.\n"
        "Action: lookup_citation\n"
        "Action Input: Art. 221 StPO\n"
        "\n"
        "Observation: Prefix 'Art. 221 StPO' matched 3 citations. Top 3 keys:\n"
        "  CITATION_KEY: Art. 221 Abs. 1 StPO\n"
        "  CITATION_KEY: Art. 221 Abs. 2 StPO\n"
        "  CITATION_KEY: Art. 221 Abs. 3 StPO\n"
        "\n"
        "Thought: Use exact key Art. 221 Abs. 1 StPO.\n"
        "Action: lookup_citation\n"
        "Action Input: Art. 221 Abs. 1 StPO\n"
        "\n"
        "Observation: [Art. 221 Abs. 1 StPO] Untersuchungshaft ist zulaessig wenn...\n"
        "\n"
        "Thought: Continue with more articles then search.\n"
        "Action: lookup_citation\n"
        "Action Input: Art. 212 Abs. 3 StPO\n"
        "\n"
        "Observation: [Art. 212 Abs. 3 StPO] Die Haft darf nicht laenger dauern...\n"
        "\n"
        "Action: lookup_citation\n"
        "Action Input: Art. 100 Abs. 1 BGG\n"
        "\n"
        "Observation: [Art. 100 Abs. 1 BGG] Die Beschwerde...\n"
        "\n"
        "Action: search_courts\n"
        "Action Input: Untersuchungshaft Kollusionsgefahr Verhaeltnismaessigkeit\n"
        "\n"
        "Observation: - BGE 137 IV 122 E. 6.2: Die Kollusionsgefahr...\n"
        "\n"
        "Action: lookup_citation\n"
        "Action Input: BGE 137 IV 122 E. 6.2\n"
        "\n"
        "Observation: [BGE 137 IV 122 E. 6.2] ...\n"
        "\n"
        "CITATIONS: BGE 143 IV 168 E. 3.1; Art. 221 Abs. 1 StPO; Art. 212 Abs. 3 StPO; Art. 100 Abs. 1 BGG; BGE 137 IV 122 E. 6.2\n"
    )

    new_prompt_section = start_marker + NEW_PROMPT_BODY + '"""'
    s = s[:prompt_start] + new_prompt_section + s[prompt_end + 3:]
    set_src(nb["cells"][prompt_idx], s)
    results.append("Fix 2+3 OK: new 5-step AGENT_SYSTEM_PROMPT with domain hints")

# ── Save ──────────────────────────────────────────────────────────────────────
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
    ("exact match capped 200",     "if len(text) > 200:" in all_src),
    ("prefix keys only",           "CITATION_KEY: {k}" in all_src),
    ("prefix no text parts",       "parts = []" not in all_src.split("class HyDESearchTool")[0].split("class CitationLookupTool")[-1]),
    ("max_prefix_results=3",       "max_prefix_results=3," in all_src),
    ("5-step prompt",              "MANDATORY 5-STEP PROCEDURE" in all_src),
    ("hyde STEP 2 mandatory",      "STEP 2 - DENSE SEMANTIC SEARCH" in all_src),
    ("IVG domain hints",           "INVALIDITY INSURANCE" in all_src),
    ("StPO domain hints",          "PRE-TRIAL DETENTION" in all_src),
    ("IVG articles",               "Art. 17 IVG" in all_src),
    ("ATSG articles",              "Art. 16 ATSG" in all_src),
]
for label, ok in checks:
    print(f"  {'OK' if ok else 'FAIL'} {label}")
