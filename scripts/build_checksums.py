"""Create a deterministic SHA-256 inventory for the compact release."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SHA256SUMS.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


files = sorted(
    path
    for path in ROOT.rglob("*")
    if path.is_file()
    and path != OUTPUT
    and ".git" not in path.parts
    and ".venv" not in path.parts
    and "__pycache__" not in path.parts
    and "replay" not in path.relative_to(ROOT).parts
)
lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(f"Wrote {len(lines)} entries to {OUTPUT.name}")

