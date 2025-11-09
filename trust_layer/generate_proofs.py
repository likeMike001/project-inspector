"""CLI utility to (re)build the trust proofs registry."""
from __future__ import annotations

from .verification import build_registry, save_registry


def main() -> None:
    registry = build_registry()
    save_registry(registry)
    print(f"Updated trust registry with {len(registry)} dataset entries.")


if __name__ == "__main__":
    main()
