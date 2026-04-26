"""
Fix citation output in both german-2 and baseline notebooks:
  1. CITATIONS: line = termination signal only, never extract citations from it
  2. Remove extract_citations_from_text fallback (source of template text junk)
  3. Add _filter_valid_citations() — strict regex validator applied to tool outputs
  4. Only tool.get_last_citations() populates the final citation list

Fixes:
  - Template text "[relevant docket number]" leaking into output
  - "[lookup_citation(...)]" pseudocode in output
  - "- Art. X SVG" with leading dash
  - Domain hints emitted without tool calls
  - Empty rows: correct behavior (no tools called = no output)
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

NOTEBOOKS = [
    "notebooks/02_agentic_retrieval_baseline_german-2.ipynb",
    "notebooks/02_agentic_retrieval_baseline.ipynb",
    "notebooks/04_hyde_agent.ipynb",
]

NEW_RUN_AGENT = '''\
def _filter_valid_citations(citations: list[str]) -> list[str]:
    """Keep only strings that look like real Swiss legal citations."""
    import re as _re
    _PATTERNS = [
        r"^Art\\.\\s+\\d+[a-z]?\\s+Abs\\.\\s+\\d+\\w*\\s+[A-Z]\\w+",  # Art. X Abs. Y LAW
        r"^Art\\.\\s+\\d+[a-z]?[a-z]*\\s+[A-Z]{2,}",                   # Art. X LAW
        r"^BGE\\s+\\d{2,3}\\s+[IVX]+\\w*\\s+\\d+",                     # BGE X IV Y ...
        r"^\\d+[A-Z]_\\d+/\\d{4}",                                      # 1B_90/2021 ...
        r"^\\d+[A-Z]{2,}_\\d+/\\d{4}",                                  # 8C_160/2016 ...
    ]
    _BAD = ["[", "]", "lookup_citation", "Note:", "CITATION_KEY",
            "\\n", "relevant", "obtained", "assuming", "actual"]
    seen, out = set(), []
    for cit in citations:
        cit = cit.strip().lstrip("- •*")
        if not cit or cit in seen:
            continue
        if any(b.lower() in cit.lower() for b in _BAD):
            continue
        if any(_re.match(p, cit) for p in _PATTERNS):
            seen.add(cit)
            out.append(cit)
    return out


def parse_all_agent_actions(response: str) -> list[tuple[str, str]]:
    import re
    actions = []
    action_matches = list(re.finditer(r"Action:\\s*(\\w+)", response, re.IGNORECASE))
    for i, am in enumerate(action_matches):
        action    = am.group(1).strip()
        start_pos = am.end()
        end_pos   = action_matches[i+1].start() if i+1 < len(action_matches) else len(response)
        m = re.search(
            r"Action Input:\\s*(.+?)(?=\\nAction:|$)",
            response[start_pos:end_pos],
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            actions.append((action, m.group(1).strip()))
    return actions


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
    sys_part = conv[:inst_end + 7]
    rest     = conv[inst_end + 7:]
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
        conversation = truncate_conversation(
            conversation, CONFIG.get("max_conversation_chars", 28000)
        )

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
                f"[INST] {AGENT_SYSTEM_PROMPT}\\n\\nQuery: {query} [/INST]"
                f"\\n\\nThought:{response}"
            )
        else:
            conversation += response

        logs.append({"type": "llm", "iteration": iteration + 1, "response": response[:500]})

        actions = parse_all_agent_actions(response)
        observations = []

        for action, action_input in actions:
            al = action.lower()
            if al in TOOLS:
                tool = TOOLS[al]
                obs  = tool(action_input)
                # Only tool.get_last_citations() adds to the citation pool
                all_citations.extend(tool.get_last_citations())
                obs_trunc = truncate_observation(obs, CONFIG.get("max_observation_chars", 1000))
                observations.append(
                    f"Observation: [{al}({action_input[:40]})] {obs_trunc}"
                )
                logs.append({
                    "type": "tool", "iteration": iteration + 1,
                    "tool": action, "query": action_input,
                    "citations": tool.get_last_citations(), "obs": obs[:300],
                })
                if verbose:
                    print(f"  [{action}({action_input[:40]})] -> {len(tool.get_last_citations())} cits")
            else:
                observations.append(
                    f"Observation: Unknown tool {action!r}. "
                    f"Available: {', '.join(TOOLS.keys())}"
                )

        if observations:
            conversation += "\\n" + "\\n".join(observations) + "\\nThought:"

        # CITATIONS: line = termination signal only — do NOT extract from it
        # (model often includes template text or unverified citations there)
        if "CITATIONS:" in response:
            break

        # No actions and no termination = model is stuck, stop
        if not actions:
            break

    # Strict filter: only emit strings that match known citation patterns
    return _filter_valid_citations(all_citations), logs


print("Agent defined. max_iterations =", CONFIG["max_iterations"])
'''

def src(c):
    s = c["source"]
    return "".join(s) if isinstance(s, list) else s

results = []

for NB_PATH in NOTEBOOKS:
    try:
        nb = json.load(open(NB_PATH, encoding='utf-8'))
    except FileNotFoundError:
        results.append(f"SKIP {NB_PATH} (not found)")
        continue

    # Find the cell containing run_agent
    agent_idx = None
    for i, c in enumerate(nb['cells']):
        if 'def run_agent(' in src(c) and 'AGENT_SYSTEM_PROMPT' in src(c):
            agent_idx = i
            break

    if agent_idx is None:
        results.append(f"SKIP {NB_PATH} — run_agent cell not found")
        continue

    s = src(nb['cells'][agent_idx])

    # Keep everything up to and including AGENT_SYSTEM_PROMPT + helper functions,
    # then replace from parse_all_agent_actions onward with our new version
    # Find the boundary: end of AGENT_SYSTEM_PROMPT closing triple-quote
    prompt_start = s.index('AGENT_SYSTEM_PROMPT = """')
    content_start = prompt_start + len('AGENT_SYSTEM_PROMPT = """')
    prompt_end = s.index('"""', content_start)
    # Keep: import re + AGENT_SYSTEM_PROMPT
    keep = s[:prompt_end + 3]  # up to and including closing """

    nb['cells'][agent_idx]['source'] = keep + "\n\n\n" + NEW_RUN_AGENT

    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    # Verify
    nb2 = json.load(open(NB_PATH, encoding='utf-8'))
    all_src = "".join(
        ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
        for c in nb2['cells']
    )
    checks = {
        "_filter_valid_citations":     "_filter_valid_citations" in all_src,
        "no extract_citations_from_text": "extract_citations_from_text" not in all_src,
        "CITATIONS termination only":  "termination signal only" in all_src,
        "tool.get_last_citations only": "Only tool.get_last_citations" in all_src,
        "BAD filter has template text": '"relevant"' in all_src,
    }
    ok = all(checks.values())
    results.append(f"{'OK' if ok else 'PARTIAL'} {NB_PATH}")
    for label, passed in checks.items():
        results.append(f"    {'OK' if passed else 'FAIL'} {label}")

print("\n".join(results))
