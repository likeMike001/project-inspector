"""Core hashing / verification helpers for the trust layer."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .config import DATASETS, PROOFS_PATH


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _simulate_eigenlayer(dataset_id: str) -> Dict[str, object]:
    return {
        "simulated": True,
        "proof_id": f"eigen-sim::{dataset_id}::{datetime.now(timezone.utc).isoformat()}",
        "confidence": 0.9,
    }


def _simulate_zkp(dataset_id: str) -> Dict[str, object]:
    return {
        "scheme": "zkp-demo",
        "status": "pass",
        "dataset": dataset_id,
    }


def build_registry() -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    verified_at = datetime.now(timezone.utc).isoformat()
    for dataset in DATASETS:
        path = dataset["path"]
        exists = path.exists()
        entry = {
            "id": dataset["id"],
            "label": dataset["label"],
            "path": str(path),
            "status": "ok" if exists else "missing",
            "size_bytes": path.stat().st_size if exists else None,
            "sha256": _sha256_file(path) if exists else None,
            "last_verified_at": verified_at,
            "eigenlayer_attestation": _simulate_eigenlayer(dataset["id"]),
            "zkp_simulation": _simulate_zkp(dataset["id"]),
        }
        entries.append(entry)
    return entries


def save_registry(entries: List[Dict[str, object]], path: Path = PROOFS_PATH) -> None:
    path.write_text(json.dumps(entries, indent=2))


def load_registry(path: Path = PROOFS_PATH) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return json.loads(path.read_text())
