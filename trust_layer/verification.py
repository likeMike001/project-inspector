"""Core hashing / verification helpers for the trust layer."""
from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .config import DATASETS, PROOFS_PATH

# Large prime from BN254; gives us a deterministic finite field without extra deps.
SCHNORR_P = 21888242871839275222246405745257275088548364400416034343698204186575808495617
SCHNORR_Q = SCHNORR_P - 1
SCHNORR_G = 5


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


def _hash_to_int(*parts: bytes) -> int:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part)
    return int.from_bytes(hasher.digest(), "big") % SCHNORR_Q


def _schnorr_proof(dataset_id: str, digest_hex: str) -> Dict[str, object]:
    secret = int(digest_hex, 16) % SCHNORR_Q
    if secret == 0:
        secret = 1
    nonce = secrets.randbelow(SCHNORR_Q)
    commitment = pow(SCHNORR_G, nonce, SCHNORR_P)
    challenge = _hash_to_int(
        dataset_id.encode("utf-8"),
        digest_hex.encode("utf-8"),
        str(commitment).encode("utf-8"),
    )
    response = (nonce + challenge * secret) % SCHNORR_Q
    public_key = pow(SCHNORR_G, secret, SCHNORR_P)
    proof = {
        "scheme": "schnorr-sha256",
        "status": "pass",
        "public_key": str(public_key),
        "commitment": str(commitment),
        "challenge": str(challenge),
        "response": str(response),
    }
    if not _verify_schnorr(dataset_id, digest_hex, proof):
        proof["status"] = "fail"
    return proof


def _verify_schnorr(dataset_id: str, digest_hex: str, proof: Dict[str, object]) -> bool:
    try:
        public_key = int(proof["public_key"])
        commitment = int(proof["commitment"])
        challenge = int(proof["challenge"])
        response = int(proof["response"])
    except (KeyError, ValueError, TypeError):
        return False

    recomputed_challenge = _hash_to_int(
        dataset_id.encode("utf-8"),
        digest_hex.encode("utf-8"),
        str(commitment).encode("utf-8"),
    )
    if challenge % SCHNORR_Q != recomputed_challenge:
        return False

    left = pow(SCHNORR_G, response, SCHNORR_P)
    right = (commitment * pow(public_key, challenge, SCHNORR_P)) % SCHNORR_P
    return left == right


def _build_zk_entry(dataset_id: str, digest_hex: Optional[str]) -> Dict[str, object]:
    if not digest_hex:
        return {
            "scheme": "schnorr-sha256",
            "status": "missing",
            "public_key": None,
            "commitment": None,
            "challenge": None,
            "response": None,
        }
    return _schnorr_proof(dataset_id, digest_hex)


def build_registry() -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    verified_at = datetime.now(timezone.utc).isoformat()
    for dataset in DATASETS:
        path = dataset["path"]
        exists = path.exists()
        digest = _sha256_file(path) if exists else None
        entry = {
            "id": dataset["id"],
            "label": dataset["label"],
            "path": str(path),
            "status": "ok" if exists else "missing",
            "size_bytes": path.stat().st_size if exists else None,
            "sha256": digest,
            "last_verified_at": verified_at,
            "eigenlayer_attestation": _simulate_eigenlayer(dataset["id"]),
            "zkp_simulation": _build_zk_entry(dataset["id"], digest),
        }
        entries.append(entry)
    return entries


def save_registry(entries: List[Dict[str, object]], path: Path = PROOFS_PATH) -> None:
    path.write_text(json.dumps(entries, indent=2))


def load_registry(path: Path = PROOFS_PATH) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return json.loads(path.read_text())
