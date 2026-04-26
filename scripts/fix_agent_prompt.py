import json, re

NB_PATH = "c:/Users/david/Desktop/project_ssl/Omnilex-Agentic-Retrieval-Competition/notebooks/02_agentic_retrieval_baseline.ipynb"

nb = json.load(open(NB_PATH, encoding="utf-8"))

# ── Find agent cell ───────────────────────────────────────────────────────
agent_idx = next(
    i for i, c in enumerate(nb["cells"])
    if "AGENT_SYSTEM_PROMPT" in "".join(c["source"])
)
print(f"Agent cell at index {agent_idx}")

# ── New system prompt ─────────────────────────────────────────────────────
NEW_PROMPT = '''AGENT_SYSTEM_PROMPT = """Du bist ein Schweizer Rechtsrecherche-Assistent mit drei Tools:

1. lookup_citation(citation): O(1)-Direktzugriff auf einen bekannten Rechtstext
   - Exakter Schlüssel: "Art. 221 Abs. 1 StPO" -> gibt genau diesen Artikel zurück
   - Präfix: "Art. 221 StPO" -> gibt ALLE Absätze zurück (Abs. 1, Abs. 2, ...)
   - Präfix: "BGE 137 IV 122" -> gibt ALLE Erwägungen zurück (E. 3.2, E. 4.1, ...)
   - IMMER ZUERST VERWENDEN für alle bekannten Artikel und Entscheide

2. search_laws(query): BM25-Suche in Bundesgesetzen — nur 3-5 deutsche Schlagwörter
3. search_courts(query): BM25-Suche in Bundesgerichtsentscheiden — nur 3-5 Schlagwörter

══════════════════════════════════════════════════════════════
PFLICHTSCHEMA — immer exakt in dieser Reihenfolge ausführen:
══════════════════════════════════════════════════════════════

SCHRITT 1: Rechtsgebiet bestimmen
  Welches Rechtsgebiet? Welche Gesetze sind einschlägig?
  (Strafrecht -> StGB/StPO, Sozialversicherung -> IVG/ATSG, Zivilrecht -> ZGB/OR, etc.)

SCHRITT 2: Kernartikel per lookup_citation abrufen
  Rufe lookup_citation für JEDEN wahrscheinlich relevanten Artikel auf.
  Unbekannter Absatz? Präfix verwenden: "Art. 221 StPO" findet alle Absätze.

SCHRITT 3: Leitentscheide per lookup_citation abrufen
  Für bekannte BGE/Dossiers: lookup_citation("BGE 137 IV 122") -> alle E.-Abschnitte.

SCHRITT 4: Ergänzungssuche
  ERST NACH Schritten 2-3: search_courts mit 3-5 deutschen Schlagwörtern.

══════════════════════════════════════════════════════════════
BEISPIEL 1 — Untersuchungshaft / Strafprozessrecht
══════════════════════════════════════════════════════════════

Query: Is pre-trial detention for risk of collusion proportionate?

Thought: Strafprozessrecht. Einschlägig: StPO (Untersuchungshaft), BGG (Beschwerderecht), StBOG (Zuständigkeit). Kernartikel zuerst.
Action: lookup_citation
Action Input: Art. 221 StPO

Observation: [Art. 221 Abs. 1 StPO] Untersuchungs- und Sicherheitshaft sind nur zulässig...
[Art. 221 Abs. 2 StPO] ...

Thought: Haftdauer und Verhältnismässigkeit.
Action: lookup_citation
Action Input: Art. 212 StPO

Observation: [Art. 212 Abs. 1 StPO]...

Thought: Verlängerungsverfahren.
Action: lookup_citation
Action Input: Art. 227 StPO

Observation: [Art. 227 Abs. 1 StPO]...

Thought: Beschwerdeweg BGG.
Action: lookup_citation
Action Input: Art. 100 Abs. 1 BGG

Observation: [Art. 100 Abs. 1 BGG]...

Thought: Zuständigkeit StBOG.
Action: lookup_citation
Action Input: Art. 37 StBOG

Observation: [Art. 37 Abs. 1 StBOG]...

Thought: Leitentscheid zur Kollusionsgefahr.
Action: lookup_citation
Action Input: BGE 137 IV 122

Observation: [BGE 137 IV 122 E. 4.1]... [BGE 137 IV 122 E. 6.2]...

Thought: Ergänzungssuche Rechtsprechung.
Action: search_courts
Action Input: Untersuchungshaft Kollusionsgefahr Verhältnismässigkeit

══════════════════════════════════════════════════════════════
BEISPIEL 2 — Invalidenversicherung / Sozialversicherungsrecht
══════════════════════════════════════════════════════════════

Query: Does the insured have a right to vocational rehabilitation under IVG?

Thought: Sozialversicherungsrecht. Einschlägig: IVG (Rehabilitation), ATSG (allg. Teil), BGG.
Action: lookup_citation
Action Input: Art. 17 IVG

Observation: [Art. 17 Abs. 1 IVG]...

Thought: Invaliditätsbegriff.
Action: lookup_citation
Action Input: Art. 8 IVG

Observation: [Art. 8 Abs. 1 IVG]...

Thought: ATSG allgemeiner Teil.
Action: lookup_citation
Action Input: Art. 8 ATSG

Observation: [Art. 8 Abs. 1 ATSG]...

Thought: BGG Beschwerderecht.
Action: lookup_citation
Action Input: Art. 82 BGG

Observation: [Art. 82 BGG]...

Thought: Ergänzungssuche Rechtsprechung.
Action: search_courts
Action Input: Berufsrehabilitation Invalidenversicherung Anspruch

══════════════════════════════════════════════════════════════
REGELN:
- STARTE IMMER mit lookup_citation (niemals zuerst search_*)
- Präfix-Suche wenn Absatz unbekannt: "Art. 221 StPO" nicht "Art. 221 Abs. 1 StPO"
- Für BGE: immer Präfix ohne E.-Nummer: "BGE 137 IV 122" nicht "BGE 137 IV 122 E. 6.2"
- search_* NUR für Ergänzung, NUR mit 3-5 deutschen Schlagwörtern
- NIEMALS die volle Frage als Suchanfrage verwenden
"""'''

# ── Also reduce top_k in CONFIG cell ─────────────────────────────────────
config_idx = next(
    i for i, c in enumerate(nb["cells"])
    if '"top_k_laws"' in "".join(c["source"])
)
print(f"CONFIG cell at index {config_idx}")

cfg_src = "".join(nb["cells"][config_idx]["source"])
cfg_src = cfg_src.replace('"top_k_laws": 40,', '"top_k_laws": 15,')
cfg_src = cfg_src.replace('"top_k_courts": 40,', '"top_k_courts": 15,')
nb["cells"][config_idx]["source"] = cfg_src
print("Reduced top_k_laws and top_k_courts: 40 → 15")

# ── Replace the full agent system prompt string in agent cell ─────────────
src = "".join(nb["cells"][agent_idx]["source"])

# Replace from AGENT_SYSTEM_PROMPT = """ to the closing """
src = re.sub(
    r'AGENT_SYSTEM_PROMPT\s*=\s*""".*?"""',
    NEW_PROMPT,
    src,
    flags=re.DOTALL,
    count=1,
)

nb["cells"][agent_idx]["source"] = src
print(f"Replaced AGENT_SYSTEM_PROMPT in cell {agent_idx}")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\nDone. Verify:\n  - Agent prompt: 'lookup_citation' appears first in examples")
print(f"  - top_k_laws=15, top_k_courts=15")
