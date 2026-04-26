import json

NB_PATH = "c:/Users/david/Desktop/project_ssl/Omnilex-Agentic-Retrieval-Competition/notebooks/02_agentic_retrieval_baseline.ipynb"

nb = json.load(open(NB_PATH, encoding="utf-8"))

# Find the cell containing run_agent
agent_cell_idx = next(
    i for i, c in enumerate(nb["cells"])
    if "def run_agent" in "".join(c["source"])
)
print(f"Found run_agent cell at index {agent_cell_idx}")

src = "".join(nb["cells"][agent_cell_idx]["source"])

# ── Fix 1: observation format ─────────────────────────────────────────────
# Old: "Tool {action_lower}: {obs_truncated}"
# New: "Observation: [{action_lower}] {obs_truncated}"
# Rationale: stop token is "Observation:" — using "Tool X:" means the LLM
# never sees the stop boundary and hallucinates its own observations.

OLD1 = 'observations.append(f"Tool {action_lower}: {obs_truncated}")'
NEW1 = 'observations.append(f"Observation: [{action_lower}({action_input[:40]})] {obs_truncated}")'

assert OLD1 in src, f"Could not find:\n{OLD1}"
src = src.replace(OLD1, NEW1, 1)
print("Fix 1 applied: observation prefix -> 'Observation:'")

# ── Fix 2: conversation continuation format ───────────────────────────────
# Old: appends observations then wraps in [INST] Continue [/INST]\n\nThought:
# New: appends observations then just \nThought:
# Rationale: the mid-loop [INST]...[/INST] creates a false instruction boundary.
# In Mistral Instruct ReAct, observations are part of the generation context,
# not new user turns. Removing the wrapper stops the model from "restarting"
# its reasoning and hallucinating fresh responses from scratch each iteration.

OLD2 = ('conversation += "\\n" + "\\n".join(observations) + '
        '"\\n\\n[INST] Continue your analysis. [/INST]\\n\\nThought:"')
NEW2 = ('conversation += "\\n" + "\\n".join(observations) + "\\nThought:"')

assert OLD2 in src, f"Could not find:\n{OLD2}"
src = src.replace(OLD2, NEW2, 1)
print("Fix 2 applied: removed [INST] Continue [/INST] mid-loop wrapper")

# ── Fix 3: also fix the tool error observation format ─────────────────────
OLD3 = 'observations.append(f"Tool {action_lower}: {error_msg}")'
NEW3 = 'observations.append(f"Observation: [{action_lower}] Error: {error_msg}")'

if OLD3 in src:
    src = src.replace(OLD3, NEW3, 1)
    print("Fix 3 applied: error observation prefix -> 'Observation:'")
else:
    print("Fix 3 skipped: error observation line not found (may already be correct)")

nb["cells"][agent_cell_idx]["source"] = src

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\nDone. Patched cell {agent_cell_idx}.")
