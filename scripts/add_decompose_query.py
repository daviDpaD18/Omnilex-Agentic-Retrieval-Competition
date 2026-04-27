"""Add decompose_query_to_subqueries function + test cell after translate cell."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

NB_PATH = "notebooks/02_agentic_retrieval_baseline_german-2.ipynb"
nb = json.load(open(NB_PATH, encoding='utf-8'))

def get_src(cell):
    s = cell['source']
    return ''.join(s) if isinstance(s, list) else s

def make_code_cell(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }

# ── CHANGE 1: add "use_query_decomposition": True to CONFIG (cell 3) ──────
src3 = get_src(nb['cells'][3])
if '"use_query_decomposition"' not in src3:
    old = '    # SAC (Summary-Augmented Chunking)\n    "use_sac": True,\n}'
    new = (
        '    # SAC (Summary-Augmented Chunking)\n'
        '    "use_sac": True,\n'
        '\n'
        '    # Query decomposition into sub-queries\n'
        '    "use_query_decomposition": True,\n'
        '}'
    )
    assert old in src3, "SAC anchor not found in CONFIG"
    src3 = src3.replace(old, new, 1)
    nb['cells'][3]['source'] = src3
    print("CHANGE 1 OK: use_query_decomposition added to CONFIG")
else:
    print("CHANGE 1 already applied")

# ── Find translate cell ─────────────────────────────────────────────────────
translate_idx = None
for i, cell in enumerate(nb['cells']):
    if 'def translate_query_to_keywords(' in get_src(cell):
        translate_idx = i
        break
assert translate_idx is not None, "translate_query_to_keywords cell not found"
print(f"  translate_query_to_keywords at cell index {translate_idx}")

# ── CHANGE 2+3: insert function + test cells if not already present ─────────
already_present = any('def decompose_query_to_subqueries(' in get_src(c) for c in nb['cells'])

if already_present:
    print("CHANGE 2+3 already applied")
    insert_at = next(
        i for i, c in enumerate(nb['cells'])
        if 'def decompose_query_to_subqueries(' in get_src(c)
    )
else:
    decompose_src = (
        "import json as _json\n"
        "\n"
        "\n"
        "def decompose_query_to_subqueries(query: str, llm) -> list[str]:\n"
        '    """\n'
        "    Decomposes an English legal query into 3-4 distinct German sub-queries,\n"
        "    each targeting a different legal element of the case.\n"
        "    Returns a list of German search strings.\n"
        "    Falls back to [translate_query_to_keywords(query, llm)] on parse failure.\n"
        '    """\n'
        "    prompt = (\n"
        '        "[INST] You are a Swiss legal search assistant. Given an English legal query, "\n'
        '        "extract 3-4 distinct German search sub-queries. Each sub-query should target "\n'
        '        "a different legal element or issue in the case. "\n'
        '        "Output ONLY a JSON array of strings, no explanation.\\n\\n"\n'
        '        "Example:\\n"\n'
        '        "Query: \\"Can a court lawfully order a three-month extension of pre-trial "\n'
        '        "detention for risk of collusion?\\"\\n"\n'
        '        "Output: [\\"Untersuchungshaft Verlängerung Kollusionsgefahr\\", "\n'
        '        "\\"Verhältnismäßigkeit Untersuchungshaft StPO\\", "\n'
        '        "\\"Haftgrund Fluchtgefahr Verdunkelungsgefahr\\", "\n'
        '        "\\"Haftprüfung Bundesgericht BGG\\"]\\n\\n"\n'
        '        f"Query: {query}\\n"\n'
        '        "Output: [/INST]"\n'
        "    )\n"
        "\n"
        "    try:\n"
        "        response = llm(\n"
        "            prompt,\n"
        "            max_tokens=150,\n"
        "            temperature=0.1,\n"
        '            stop=["</s>", "[INST]"],\n'
        "            echo=False,\n"
        "        )\n"
        '        raw = response["choices"][0]["text"].strip()\n'
        '        start = raw.find("[")\n'
        '        end   = raw.rfind("]") + 1\n'
        "        if start != -1 and end > start:\n"
        "            subqueries = _json.loads(raw[start:end])\n"
        "            if isinstance(subqueries, list) and subqueries:\n"
        "                return [str(s).strip() for s in subqueries if str(s).strip()]\n"
        "    except Exception:\n"
        "        pass\n"
        "    return [translate_query_to_keywords(query, llm)]\n"
    )

    test_src = (
        "# Test decompose_query_to_subqueries\n"
        'print("Testing Query Decomposer:\\n" + "="*50)\n'
        "for q in test_queries:\n"
        '    print(f"Query: {q}")\n'
        "    subs = decompose_query_to_subqueries(q, llm)\n"
        "    for i, s in enumerate(subs, 1):\n"
        '        print(f"  {i}. {s}")\n'
        '    print("-" * 50)\n'
    )

    insert_at = translate_idx + 1
    nb['cells'].insert(insert_at,     make_code_cell(decompose_src))
    nb['cells'].insert(insert_at + 1, make_code_cell(test_src))
    print(f"CHANGE 2+3 OK: inserted at indices {insert_at} and {insert_at + 1}")

# ── Save ───────────────────────────────────────────────────────────────────
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("\nNotebook saved.")

# ── Verify ─────────────────────────────────────────────────────────────────
nb2 = json.load(open(NB_PATH, encoding='utf-8'))

def gs(c):
    s = c['source']
    return ''.join(s) if isinstance(s, list) else s

s3    = gs(nb2['cells'][3])
fn_c  = gs(nb2['cells'][insert_at])
tst_c = gs(nb2['cells'][insert_at + 1])

checks = {
    '"use_query_decomposition": True in CONFIG':    '"use_query_decomposition": True,' in s3,
    'decompose fn defined in fn cell':              'def decompose_query_to_subqueries(' in fn_c,
    'json import in fn cell':                       'import json as _json' in fn_c,
    '[INST] prompt in fn':                          '[INST]' in fn_c,
    'JSON parse in fn':                             '_json.loads(' in fn_c,
    'fallback to translate_query_to_keywords':      'translate_query_to_keywords' in fn_c,
    'max_tokens=150':                               'max_tokens=150' in fn_c,
    'temperature=0.1':                              'temperature=0.1' in fn_c,
    'test cell calls decompose_query_to_subqueries': 'decompose_query_to_subqueries' in tst_c,
    'test cell uses test_queries variable':         'test_queries' in tst_c,
    'translate cell unchanged':                     'def translate_query_to_keywords(' in gs(nb2['cells'][translate_idx]),
}
all_ok = True
print()
for k, v in checks.items():
    print(f"  {'OK' if v else 'FAIL'} {k}")
    if not v: all_ok = False
print()
print("All checks passed!" if all_ok else "Some checks FAILED.")
