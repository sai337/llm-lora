from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing config: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))
