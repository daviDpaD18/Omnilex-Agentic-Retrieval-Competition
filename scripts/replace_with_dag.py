"""Replace section 5 (ReAct agent) with DAG pipeline and update predictions cell."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

NB_PATH = "notebooks/02_agentic_retrieval_baseline_german-2.ipynb"
nb = json.load(open(NB_PATH, encoding='utf-8'))

def gs(c):
    s = c['source']
    return ''.join(s) if isinstance(s, list) else s

def make_code_cell(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

def make_md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

# ── Locate cells ─────────────────────────────────────────────────────────────
section5_md_idx = next(i for i, c in enumerate(nb['cells'])
                       if c['cell_type'] == 'markdown' and '## 5.' in gs(c))
# Section 5 spans: markdown + code (agent) + debug code  →  3 cells
# Confirmed: 20=md, 21=agent code, 22=debug code
# Verify cell 22 is the debug cell (contains run_agent call)
assert 'run_agent' in gs(nb['cells'][section5_md_idx + 2]), \
    f"Expected debug cell at {section5_md_idx+2}"
section5_cells = [section5_md_idx, section5_md_idx + 1, section5_md_idx + 2]
print(f"Section 5 cells to replace: {section5_cells}")

# Locate predictions cell (contains 'Running agent')
pred_idx = next(i for i, c in enumerate(nb['cells']) if 'Running agent' in gs(c))
print(f"Predictions cell at index: {pred_idx}")

# ── Build new section 5: markdown + DAG code ─────────────────────────────────
new_md = "## 5. Define DAG Pipeline"

# We copy _filter_valid_citations verbatim from the existing agent cell
agent_src = gs(nb['cells'][section5_md_idx + 1])
# Extract _filter_valid_citations (from its def to the next top-level def)
fc_start = agent_src.index('def _filter_valid_citations(')
fc_end   = agent_src.index('\ndef parse_all_agent_actions(')
filter_fn_src = agent_src[fc_start:fc_end].rstrip()

dag_src = (
    filter_fn_src
    + "\n\n\n"
    + '''\
DOMAIN_STATUTE_MAP = {
    "detention": [
        "Art. 221 Abs. 1 StPO", "Art. 221 Abs. 2 StPO", "Art. 212 Abs. 3 StPO",
        "Art. 226 Abs. 1 StPO", "Art. 227 Abs. 1 StPO", "Art. 228 Abs. 1 StPO",
        "Art. 231 Abs. 1 StPO", "Art. 237 Abs. 1 StPO", "Art. 100 Abs. 1 BGG",
    ],
    "disability": [
        "Art. 17 Abs. 1 IVG", "Art. 8 Abs. 1 IVG", "Art. 4 Abs. 1 IVG",
        "Art. 16 ATSG", "Art. 7 Abs. 1 ATSG", "Art. 6 ATSG", "Art. 28 Abs. 1 IVG",
    ],
    "contract": [
        "Art. 1 Abs. 1 OR", "Art. 18 Abs. 1 OR", "Art. 97 Abs. 1 OR", "Art. 41 Abs. 1 OR",
    ],
    "criminal": [
        "Art. 10 Abs. 2 StGB", "Art. 47 Abs. 1 StGB", "Art. 49 Abs. 1 StGB",
    ],
}

DOMAIN_KEYWORDS = {
    "detention": ["detention", "pre-trial", "remand", "arrest", "custody",
                  "collusion", "flight risk"],
    "disability": ["disability", "invalidity", "incapacity", "IV", "ATSG",
                   "insurance benefit", "work capacity"],
    "contract":  ["contract", "agreement", "breach", "liability", "damages", "obligation"],
    "criminal":  ["criminal", "penalty", "sentence", "offence", "conviction"],
}


def run_dag_pipeline(query, llm, embed_model, lance_table, tools, config):
    """DAG-structured legal citation retrieval pipeline.

    Steps:
      1. Domain routing (Python only)
      2. Parallel retrieval: statute lookup + BM25 + HyDE dense
      3. RRF fusion
      4. Candidate citation collection
      5. Single LLM synthesis call
      6. Filter and return validated citations
    """
    # ── Step 1: Domain routing ────────────────────────────────────────────────
    q_lower = query.lower()
    detected_domains = [
        d for d, kws in DOMAIN_KEYWORDS.items()
        if any(kw in q_lower for kw in kws)
    ]
    detected_domain = detected_domains[0] if detected_domains else None

    statute_keys: list[str] = []
    for d in detected_domains:
        for k in DOMAIN_STATUTE_MAP.get(d, []):
            if k not in statute_keys:
                statute_keys.append(k)

    # ── Step 2a: Direct statute lookup ───────────────────────────────────────
    statute_citations: list[str] = []
    lookup_tool = tools.get("lookup_citation")
    if lookup_tool:
        for key in statute_keys:
            try:
                obs = lookup_tool(key)
                if obs and "not found" not in obs.lower() and "error" not in obs.lower():
                    statute_citations.append(key)
            except Exception:
                pass

    # ── Step 2b: Sub-query BM25 search ───────────────────────────────────────
    if config.get("use_query_decomposition"):
        sub_queries = decompose_query_to_subqueries(query, llm)
    else:
        sub_queries = [translate_query_to_keywords(query, llm)]

    bm25_law_results:   list[dict] = []
    bm25_court_results: list[dict] = []
    for subq in sub_queries:
        if subq.strip():
            bm25_law_results.extend(
                laws_index.search(subq, top_k=config.get("top_k_laws", 15))
            )
            bm25_court_results.extend(
                courts_index.search(subq, top_k=config.get("top_k_courts", 15))
            )

    # ── Step 2c: HyDE dense search ────────────────────────────────────────────
    dense_results: list[dict] = []
    try:
        hyde_instance = tools.get("hyde_search_courts")
        if hyde_instance is not None:
            for subq in sub_queries:
                if not subq.strip():
                    continue
                hyp = hyde_instance._generate_hypothetical(subq)
                if not hyp:
                    continue
                vec = embed_model.encode(
                    ["query: " + hyp],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0].tolist()
                df = (
                    lance_table.search(vec)
                    .nprobes(config.get("nprobes", 20))
                    .limit(config.get("top_k_hyde", 20))
                    .to_pandas()
                )
                if not df.empty:
                    dense_results.extend(df.to_dict("records"))
    except Exception:
        pass

    # ── Step 3: RRF fusion ────────────────────────────────────────────────────
    fused_courts = reciprocal_rank_fusion(bm25_court_results, dense_results, rrf_k=60)[:30]
    fused_laws   = reciprocal_rank_fusion(bm25_law_results,   [],            rrf_k=60)[:15]

    # ── Step 4: Collect all candidate citations ───────────────────────────────
    seen_cands: set[str] = set()
    all_candidates: list[str] = []
    for c in statute_citations:
        if c not in seen_cands:
            seen_cands.add(c); all_candidates.append(c)
    for doc in fused_laws + fused_courts:
        c = doc.get("citation", "")
        if c and c not in seen_cands:
            seen_cands.add(c); all_candidates.append(c)

    # ── Step 5: Synthesis LLM call ────────────────────────────────────────────
    ctx_parts: list[str] = []
    for doc in fused_courts[:15]:
        cit  = doc.get("citation", "")
        text = doc.get("text", "")[:300]
        if cit:
            ctx_parts.append(f"[CITATION_KEY: {cit}]\\n{text}")
    for doc in fused_laws[:10]:
        cit  = doc.get("citation", "")
        text = doc.get("text", "")[:300]
        if cit:
            ctx_parts.append(f"[CITATION_KEY: {cit}]\\n{text}")
    context = "\\n\\n".join(ctx_parts)

    statute_str = "; ".join(statute_citations) if statute_citations else "none"
    synth_prompt = (
        "[INST] You are a Swiss legal citation expert. Based on the retrieved legal "
        "documents below, list ALL relevant citations for this case.\\n\\n"
        f"Query: {query}\\n\\n"
        f"Retrieved documents:\\n{context}\\n\\n"
        f"Domain statutes already identified: {statute_str}\\n\\n"
        "Output ONLY a line starting with CITATIONS: followed by all citations "
        "separated by semicolons. Include both statute articles and court decisions.\\n"
        "CITATIONS: [/INST]"
    )

    llm_citations: list[str] = []
    try:
        synth_resp = llm(
            synth_prompt,
            max_tokens=300,
            temperature=0.1,
            stop=["</s>", "[INST]", "\\n\\n"],
            echo=False,
        )["choices"][0]["text"].strip()
        if "CITATIONS:" in synth_resp:
            cit_line = synth_resp.split("CITATIONS:", 1)[1].split("\\n")[0].strip()
        else:
            cit_line = synth_resp
        llm_citations = [p.strip() for p in cit_line.split(";") if p.strip()]
    except Exception:
        pass

    # ── Step 6: Filter and return ─────────────────────────────────────────────
    seen_final: set[str] = set()
    final_raw: list[str] = []
    for c in statute_citations + llm_citations + all_candidates:
        if c not in seen_final:
            seen_final.add(c); final_raw.append(c)

    validated = _filter_valid_citations(final_raw)

    print(f"  [DAG] domain={detected_domain or 'none':12s} | "
          f"sub_queries={len(sub_queries)} | citations={len(validated)}")

    return {"citations": validated, "sub_queries": sub_queries, "domain": detected_domain}


print("DAG pipeline defined. DOMAIN_STATUTE_MAP domains:", list(DOMAIN_STATUTE_MAP.keys()))
'''
)

# ── Replace cells 20, 21, 22 with new markdown + code ────────────────────────
del nb['cells'][section5_cells[0]:section5_cells[-1] + 1]
nb['cells'].insert(section5_cells[0],     make_md_cell(new_md))
nb['cells'].insert(section5_cells[0] + 1, make_code_cell(dag_src))
print(f"Replaced section 5 cells with 2 new cells at indices {section5_cells[0]}, {section5_cells[0]+1}")

# ── Update predictions cell ──────────────────────────────────────────────────
# Re-find predictions cell after the splice above (index shifted by -1)
pred_idx2 = next(i for i, c in enumerate(nb['cells']) if 'Running agent' in gs(c))
print(f"Predictions cell now at index: {pred_idx2}")

old_pred_src = gs(nb['cells'][pred_idx2])

OLD_CALL = (
    "    citations, logs = run_agent(query, verbose=False)\n"
    "\n"
    "    predictions.append({\n"
    '        "query_id":            qid,\n'
    '        "predicted_citations": ";".join(citations),\n'
    "    })\n"
    '    all_logs.append({"query_id": qid, "query": query, "logs": logs})'
)
NEW_CALL = (
    "    result = run_dag_pipeline(\n"
    "        query=query,\n"
    "        llm=llm,\n"
    "        embed_model=_embed_model,\n"
    "        lance_table=_lance_table,\n"
    "        tools=TOOLS,\n"
    "        config=CONFIG,\n"
    "    )\n"
    "    raw_citations = result[\"citations\"]\n"
    "    citations = _filter_valid_citations(raw_citations)\n"
    "\n"
    "    predictions.append({\n"
    '        "query_id":            qid,\n'
    '        "predicted_citations": ";".join(citations),\n'
    "    })\n"
    '    all_logs.append({"query_id": qid, "query": query, "logs": result})'
)

# Also update the tqdm desc from "Running agent" to "Running DAG pipeline"
OLD_DESC = 'desc="Running agent"'
NEW_DESC = 'desc="Running DAG pipeline"'

assert OLD_CALL in old_pred_src, f"Old call not found in predictions cell:\n{old_pred_src}"
assert OLD_DESC in old_pred_src

new_pred_src = old_pred_src.replace(OLD_CALL, NEW_CALL, 1).replace(OLD_DESC, NEW_DESC, 1)
nb['cells'][pred_idx2]['source'] = new_pred_src
print("Predictions cell updated.")

# ── Save ─────────────────────────────────────────────────────────────────────
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("\nNotebook saved.")

# ── Verify ───────────────────────────────────────────────────────────────────
nb2  = json.load(open(NB_PATH, encoding='utf-8'))
s_md = gs(nb2['cells'][section5_cells[0]])
s_fn = gs(nb2['cells'][section5_cells[0] + 1])
s_pr = gs(nb2['cells'][pred_idx2])

checks = {
    'new markdown is ## 5. Define DAG Pipeline':    '## 5. Define DAG Pipeline' in s_md,
    '_filter_valid_citations in dag cell':          'def _filter_valid_citations(' in s_fn,
    'DOMAIN_STATUTE_MAP in dag cell':               'DOMAIN_STATUTE_MAP' in s_fn,
    'DOMAIN_KEYWORDS in dag cell':                  'DOMAIN_KEYWORDS' in s_fn,
    'run_dag_pipeline defined':                     'def run_dag_pipeline(' in s_fn,
    'Step 1 domain routing':                        'detected_domains' in s_fn,
    'Step 2a statute lookup':                       'statute_citations' in s_fn,
    'Step 2b BM25 sub-queries':                     'bm25_law_results' in s_fn,
    'Step 2c HyDE dense':                           '_generate_hypothetical' in s_fn,
    'Step 3 RRF fusion':                            'fused_courts' in s_fn,
    'Step 5 LLM synthesis':                         'synth_prompt' in s_fn,
    'max_tokens=300':                               'max_tokens=300' in s_fn,
    'Step 6 filter + return dict':                  'return {"citations"' in s_fn,
    'predictions cell calls run_dag_pipeline':      'run_dag_pipeline(' in s_pr,
    'predictions cell has raw_citations':           'raw_citations' in s_pr,
    'predictions cell calls _filter_valid_citations': '_filter_valid_citations(raw_citations)' in s_pr,
    'old run_agent call removed from pred cell':    'run_agent(' not in s_pr,
    'Running DAG pipeline desc':                    'Running DAG pipeline' in s_pr,
}
all_ok = True
print()
for k, v in checks.items():
    print(f"  {'OK' if v else 'FAIL'} {k}")
    if not v: all_ok = False
print()
print("All checks passed!" if all_ok else "Some checks FAILED.")
