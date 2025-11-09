"""Static configuration for the Trust Verification Layer."""
from __future__ import annotations

from pathlib import Path
from typing import List, TypedDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROOFS_PATH = Path(__file__).with_name("trust_proofs.json")


class DatasetEntry(TypedDict):
    id: str
    label: str
    path: Path


DATASETS: List[DatasetEntry] = [
    {
        "id": "etherfi_combined_labeled",
        "label": "EtherFi Combined Labeled Dataset",
        "path": PROJECT_ROOT / "etherfi_combined_labeled.csv",
    },
    {
        "id": "etherfi_combined_raw",
        "label": "EtherFi Combined (Raw)",
        "path": PROJECT_ROOT / "etherfi_combined.csv",
    },
    {
        "id": "final_with_sentiment",
        "label": "Final Dataset with Sentiment",
        "path": PROJECT_ROOT / "data" / "final_with_sentiment.csv",
    },
    {
        "id": "synthetic_news",
        "label": "Synthetic News Export",
        "path": PROJECT_ROOT / "data" / "synthetic_news_string_20251109_060057.csv",
    },
    {
        "id": "eeth_apr",
        "label": "eETH APR Snapshot",
        "path": PROJECT_ROOT / "datasets" / "eETH_APR.csv",
    },
    {
        "id": "eeth_active_holders",
        "label": "eETH Active Holder Snapshot",
        "path": PROJECT_ROOT / "datasets" / "eETH_Active_Holder.csv",
    },
    {
        "id": "holder_retention",
        "label": "EtherFi Holder Retention",
        "path": PROJECT_ROOT / "datasets" / "etherFI_Holder_retention.csv",
    },
    {
        "id": "deposit_retention",
        "label": "EtherFi Deposit Retention",
        "path": PROJECT_ROOT / "datasets" / "etherFI_deposit_Retention.csv",
    },
    {
        "id": "vault_deposit_retention",
        "label": "EtherFi Liquid Vault Deposit Retention",
        "path": PROJECT_ROOT / "datasets" / "etherFI_Liquid_Valut_Deposit_retention.csv",
    },
]
