"""
Three targeted fixes to 04_hyde_agent.ipynb:
  Fix A: max_iterations 6 -> 10
  Fix B: HyDE distance threshold 0.50 -> 0.30, hard cap at 5 results
  Fix C: CitationLookupTool law-code-aware prefix matching
"""
import json, re

NB_PATH = "notebooks/04_hyde_agent.ipynb"
nb = json.load(open(NB_PATH, encoding="utf-8"))

def src(c): s = c["source"]; return "".join(s) if isinstance(s, list) else s
def set_src(c, s): c["source"] = s

results = []

# ── Fix A: max_iterations 6 -> 10 ────────────────────────────────────────
config_idx = 3
s = src(nb["cells"][config_idx])
if '"max_iterations": 6,' in s:
    s = s.replace('"max_iterations": 6,', '"max_iterations": 10,', 1)
    set_src(nb["cells"][config_idx], s)
    results.append("Fix A OK: max_iterations 6 -> 10")
else:
    results.append("Fix A FAIL: max_iterations=6 not found")

# ── Fix B: HyDE tighter threshold + hard cap ─────────────────────────────
tools_idx = 11
s = src(nb["cells"][tools_idx])

OLD_DIST = (
    "        # Step 4: filter by cosine distance threshold\n"
    "        # LanceDB returns _distance: lower = more similar (cosine)\n"
    "        DIST_THRESHOLD = 0.50\n"
    "        top = [c for c in candidates if c.get(\"_distance\", 1.0) < DIST_THRESHOLD]\n"
    "        if not top:\n"
    "            top = candidates[:3]  # always return at least 3\n"
)

NEW_DIST = (
    "        # Step 4: filter by cosine distance threshold + hard cap\n"
    "        # LanceDB returns _distance: lower = more similar (cosine)\n"
    "        DIST_THRESHOLD = 0.30\n"
    "        top = [c for c in candidates if c.get(\"_distance\", 1.0) < DIST_THRESHOLD]\n"
    "        top = top[:5]  # hard cap: never flood context with >5 results\n"
    "        if not top:\n"
    "            top = candidates[:2]  # always return at least 2\n"
)

if OLD_DIST in s:
    s = s.replace(OLD_DIST, NEW_DIST, 1)
    results.append("Fix B OK: DIST_THRESHOLD 0.50->0.30, hard cap at 5")
else:
    results.append("Fix B FAIL: distance block not found")

# ── Fix C: Law-code-aware prefix matching in CitationLookupTool ───────────
# Replace the prefix matching comment + candidates block with smarter version

OLD_PREFIX = (
    "        # 2. Prefix match � e.g. \"Art. 221 StPO\" finds \"Art. 221 Abs. 1 StPO\" etc.\n"
    "        # Also handles \"BGE 137 IV 122\" finding all E. sections\n"
    "        candidates = []\n"
    "        for store in [self._laws, self._courts]:\n"
    "            candidates += [\n"
    "                (k, v) for k, v in store.items()\n"
    "                if k.startswith(citation)\n"
    "            ]\n"
)

# Try exact match first, then try with mojibake character
OLD_PREFIX_ALT = (
    "        # 2. Prefix match � e.g. \"Art. 221 StPO\" finds \"Art. 221 Abs. 1 StPO\" etc.\n"
    "        # Also handles \"BGE 137 IV 122\" finding all E. sections\n"
    "        candidates = []\n"
    "        for store in [self._laws, self._courts]:\n"
    "            candidates += [\n"
    "                (k, v) for k, v in store.items()\n"
    "                if k.startswith(citation)\n"
    "            ]\n"
)

NEW_PREFIX = (
    "        # 2. Prefix match — e.g. \"Art. 221 StPO\" finds \"Art. 221 Abs. 1 StPO\" etc.\n"
    "        # Also handles \"BGE 137 IV 122\" finding all E. sections.\n"
    "        # Law-code-aware: \"Art. 221 StPO\" splits into prefix=\"Art. 221\" + law=\"StPO\"\n"
    "        # and finds any key starting with \"Art. 221\" that also contains \"StPO\".\n"
    "        import re as _re2\n"
    "        _art_law = _re2.match(\n"
    "            r'^(Art\\.\\s+\\d+\\w*)\\s+([A-Z]{2,}(?:bis|ter)?)$', citation\n"
    "        )\n"
    "        candidates = []\n"
    "        for store in [self._laws, self._courts]:\n"
    "            if _art_law:\n"
    "                art_pfx = _art_law.group(1)  # e.g. \"Art. 221\"\n"
    "                law_sfx = _art_law.group(2)  # e.g. \"StPO\"\n"
    "                candidates += [\n"
    "                    (k, v) for k, v in store.items()\n"
    "                    if k.startswith(art_pfx) and law_sfx in k\n"
    "                ]\n"
    "            else:\n"
    "                candidates += [\n"
    "                    (k, v) for k, v in store.items()\n"
    "                    if k.startswith(citation)\n"
    "                ]\n"
)

# Find the prefix block by searching for the key pattern regardless of encoding
idx = s.find("        # 2. Prefix match")
if idx != -1:
    # Find end of the candidates block (the closing bracket + newline)
    end_marker = "                if k.startswith(citation)\n            ]\n"
    end_idx = s.find(end_marker, idx)
    if end_idx != -1:
        old_block = s[idx:end_idx + len(end_marker)]
        s = s[:idx] + NEW_PREFIX + s[end_idx + len(end_marker):]
        results.append("Fix C OK: law-code-aware prefix matching added")
    else:
        results.append("Fix C FAIL: end of prefix block not found")
else:
    results.append("Fix C FAIL: prefix comment not found")

set_src(nb["cells"][tools_idx], s)

# ── Save ──────────────────────────────────────────────────────────────────
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
    ("max_iterations=10",         '"max_iterations": 10,' in all_src),
    ("DIST_THRESHOLD=0.30",       "DIST_THRESHOLD = 0.30" in all_src),
    ("hard cap at 5",             "top[:5]" in all_src),
    ("law-code-aware prefix",     "_art_law" in all_src),
    ("art_pfx + law_sfx search",  "law_sfx in k" in all_src),
]
for label, ok in checks:
    print(f"  {'OK' if ok else 'FAIL'} {label}")

# Smoke test the prefix logic
print("\nSmoke test prefix matching logic:")
test_cases = [
    ("Art. 221 StPO",     True,  "should match Art. 221 Abs. 1 StPO"),
    ("Art. 17 IVG",       True,  "should match Art. 17 Abs. 1 IVG"),
    ("Art. 100 BGG",      True,  "should match Art. 100 Abs. 1 BGG"),
    ("BGE 137 IV 122",    False, "should use plain startswith"),
    ("1B_90/2021",        False, "should use plain startswith"),
]
import re
for cit, expect_art, note in test_cases:
    m = re.match(r'^(Art\.\s+\d+\w*)\s+([A-Z]{2,}(?:bis|ter)?)$', cit)
    got_art = m is not None
    status = "OK" if got_art == expect_art else "FAIL"
    pfx = f"art_pfx={m.group(1)!r} law={m.group(2)!r}" if m else "plain prefix"
    print(f"  {status} {cit!r:30} -> {pfx}  ({note})")
