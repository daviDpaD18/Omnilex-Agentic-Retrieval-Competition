"""Apply SAC (Summary-Augmented Chunking) changes to the notebook."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

NB_PATH = "notebooks/02_agentic_retrieval_baseline_german-2.ipynb"
nb = json.load(open(NB_PATH, encoding='utf-8'))

def get_src(cell):
    s = cell['source']
    return ''.join(s) if isinstance(s, list) else s

def set_src(cell, src):
    cell['source'] = src

src3 = get_src(nb['cells'][3])
src5 = get_src(nb['cells'][5])
src7 = get_src(nb['cells'][7])

# ── CHANGE 1: add "use_sac": True to CONFIG (cell 3) ──────────────────────
if '"use_sac": True' not in src3:
    old = '    # Paths\n    "test_file": "test.csv",\n}'
    new = '    # Paths\n    "test_file": "test.csv",\n\n    # SAC (Summary-Augmented Chunking)\n    "use_sac": True,\n}'
    assert old in src3, "CONFIG end anchor not found"
    src3 = src3.replace(old, new, 1)
    set_src(nb['cells'][3], src3)
    print("CHANGE 1 OK: use_sac added to CONFIG")
else:
    print("CHANGE 1 already applied")

# ── CHANGE 2a: insert build_sac_documents before get_or_build_index ────────
if 'def build_sac_documents(' not in src5:
    sac_fn = (
        "def build_sac_documents(documents: list[dict]) -> list[dict]:\n"
        '    """Summary-Augmented Chunking: prepend a 150-char decision summary to every chunk.\n'
        "\n"
        "    Groups rows sharing the same base decision (same docket/BGE prefix before\n"
        "    ' E.') into a group, builds a 150-char summary from concatenated group text,\n"
        "    and prepends [KONTEXT: <summary>] to each chunk's text field.\n"
        "    Only intended for the courts corpus.\n"
        '    """\n'
        "    import re as _re_sac\n"
        "    from collections import defaultdict\n"
        "\n"
        "    def _base_citation(cit: str) -> str:\n"
        "        m = _re_sac.match(r'^(.+?)\\s+E\\.\\s+', cit)\n"
        "        return m.group(1) if m else cit\n"
        "\n"
        "    groups: dict[str, list[int]] = defaultdict(list)\n"
        "    for idx, doc in enumerate(documents):\n"
        "        groups[_base_citation(doc.get('citation', ''))].append(idx)\n"
        "\n"
        "    n_groups = len(groups)\n"
        "    result = [doc.copy() for doc in documents]\n"
        "    for base, indices in groups.items():\n"
        "        full_text = ' '.join(documents[i].get('text', '') for i in indices)\n"
        "        summary = ' '.join(full_text.split())[:150]\n"
        "        prefix = '[KONTEXT: ' + summary + '] '\n"
        "        for i in indices:\n"
        "            result[i]['text'] = prefix + result[i]['text']\n"
        "\n"
        "    print(f'  SAC: {n_groups:,} unique decision groups across {len(documents):,} chunks')\n"
        "    return result\n"
        "\n"
        "\n"
        "def get_or_build_index("
    )
    assert "def get_or_build_index(" in src5
    src5 = src5.replace("def get_or_build_index(", sac_fn, 1)
    print("CHANGE 2a OK: build_sac_documents inserted")
else:
    print("CHANGE 2a already applied")

# ── CHANGE 2b: add preprocessor param to get_or_build_index signature ──────
if 'preprocessor=None,' not in src5:
    old_sig = (
        "def get_or_build_index(\n"
        "    name: str,\n"
        "    csv_path: Path,\n"
        "    index_path: Path,\n"
        "    force_rebuild: bool = False,\n"
        "    max_rows: int | None = None\n"
        ") -> BM25Index:"
    )
    new_sig = (
        "def get_or_build_index(\n"
        "    name: str,\n"
        "    csv_path: Path,\n"
        "    index_path: Path,\n"
        "    force_rebuild: bool = False,\n"
        "    max_rows: int | None = None,\n"
        "    preprocessor=None,\n"
        ") -> BM25Index:"
    )
    assert old_sig in src5, "signature not found in cell 5"
    src5 = src5.replace(old_sig, new_sig, 1)
    print("CHANGE 2b OK: preprocessor param added")
else:
    print("CHANGE 2b already applied")

# ── CHANGE 2c: insert preprocessor call before BM25 build ──────────────────
if 'preprocessor.__name__' not in src5:
    # Find the line by searching for its start
    marker = 'Building BM25 index for'
    assert marker in src5, "BM25 build marker not found"
    pos = src5.index(marker)
    # Walk back to start of the print( call
    line_start = src5.rindex('\n', 0, pos) + 1
    line_end   = src5.index('\n', pos)
    build_line = src5[line_start:line_end]
    print(f"  Found build line: {build_line!r}")
    insert_block = (
        "    if preprocessor is not None:\n"
        "        print(f'  Applying {preprocessor.__name__} preprocessing...')\n"
        "        documents = preprocessor(documents)\n"
        "\n"
    )
    src5 = src5[:line_start] + insert_block + src5[line_start:]
    print("CHANGE 2c OK: preprocessor call inserted")
else:
    print("CHANGE 2c already applied")

set_src(nb['cells'][5], src5)

# ── CHANGE 3: pass preprocessor to courts index call (cell 7) ──────────────
if 'preprocessor=build_sac_documents' not in src7:
    old_c = (
        "    force_rebuild=FORCE_REBUILD_INDICES,\n"
        "    # max_rows=100000  # Uncomment to test with smaller corpus\n"
        ")"
    )
    new_c = (
        "    force_rebuild=FORCE_REBUILD_INDICES,\n"
        "    preprocessor=build_sac_documents if CONFIG.get(\"use_sac\") else None,\n"
        "    # max_rows=100000  # Uncomment to test with smaller corpus\n"
        ")"
    )
    assert old_c in src7, f"courts call pattern not found"
    src7 = src7.replace(old_c, new_c, 1)
    set_src(nb['cells'][7], src7)
    print("CHANGE 3 OK: preprocessor passed to courts index call")
else:
    print("CHANGE 3 already applied")

# ── Save ───────────────────────────────────────────────────────────────────
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("\nNotebook saved.")

# ── Verify ─────────────────────────────────────────────────────────────────
nb2 = json.load(open(NB_PATH, encoding='utf-8'))
s3 = ''.join(nb2['cells'][3]['source']) if isinstance(nb2['cells'][3]['source'], list) else nb2['cells'][3]['source']
s5 = ''.join(nb2['cells'][5]['source']) if isinstance(nb2['cells'][5]['source'], list) else nb2['cells'][5]['source']
s6 = ''.join(nb2['cells'][6]['source']) if isinstance(nb2['cells'][6]['source'], list) else nb2['cells'][6]['source']
s7 = ''.join(nb2['cells'][7]['source']) if isinstance(nb2['cells'][7]['source'], list) else nb2['cells'][7]['source']

checks = {
    '"use_sac": True in CONFIG':            '"use_sac": True,' in s3,
    'build_sac_documents defined':          'def build_sac_documents(' in s5,
    '_base_citation helper in fn':          '_base_citation' in s5,
    'defaultdict import in fn':             'from collections import defaultdict' in s5,
    'preprocessor param in signature':      'preprocessor=None,' in s5,
    'preprocessor call in body':            'preprocessor.__name__' in s5,
    'SAC fn defined before get_or_build':   s5.index('build_sac_documents') < s5.index('get_or_build_index'),
    'preprocessor in courts call':          'preprocessor=build_sac_documents' in s7,
    'CONFIG.get use_sac guard':             'CONFIG.get("use_sac")' in s7,
    'laws cell unchanged':                  'preprocessor' not in s6,
}
all_ok = True
print()
for k, v in checks.items():
    print(f"  {'OK' if v else 'FAIL'} {k}")
    if not v: all_ok = False
print()
print("All checks passed!" if all_ok else "Some checks FAILED.")
