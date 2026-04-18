import shutil
from pathlib import Path

import kagglehub

path = kagglehub.competition_download('llm-agentic-legal-information-retrieval')

dest = Path(__file__).resolve().parent.parent / "data"
dest.mkdir(parents=True, exist_ok=True)

for f in Path(path).iterdir():
    shutil.copy2(f, dest / f.name)

print(f"Competition files copied to: {dest}")
