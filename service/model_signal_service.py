"""
FastAPI service that exposes pickled staking models (RandomForest + clustering).

POST /signals expects a price/sentiment weighting (from the front-end slider) and
returns:
    - ranked staking actions from the RandomForest classifier
    - the active EtherFi cluster regime with contextual notes
These outputs let the UI display actionable cards whenever the slider moves.
"""

from __future__ import annotations

import pickle
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.preprocessing import LabelEncoder

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - optional dependency
    Anthropic = None

try:
    from wallet_snapshot import snapshot_wallet as wallet_snapshot_fetch
except Exception as exc:  # pragma: no cover - wallet module optional at runtime
    wallet_snapshot_fetch = None
    WALLET_IMPORT_ERROR = str(exc)
else:
    WALLET_IMPORT_ERROR = None

from model.random_forest_model import (
    FocusConfig,
    StakingSignalTrainer,
    WORKSPACE_ROOT as MODEL_ROOT,
)
from model.clustering import ClusterArtifacts

# Some pickles were produced via `python project-one/model/clustering.py` which
# registers ClusterArtifacts under `__main__`. Ensure that alias exists so those
# artifacts can still be deserialised without regeneration.
main_module = sys.modules.get("__main__")
if main_module and not hasattr(main_module, "ClusterArtifacts"):
    setattr(main_module, "ClusterArtifacts", ClusterArtifacts)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
RF_ARTIFACT_PATH = ARTIFACT_DIR / "random_forest_model.pkl"
CLUSTER_ARTIFACT_PATH = ARTIFACT_DIR / "etherfi_clusterer.pkl"
CLUSTERED_ROWS_PATH = ARTIFACT_DIR / "etherfi_clusters.pkl"
ETHERFI_SOURCE = PROJECT_ROOT / "etherfi_combined_labeled.csv"

if load_dotenv is not None:
    # Load project-level secrets when running via `uvicorn service...`
    load_dotenv(PROJECT_ROOT / ".env", override=False)

_wallet_lookback_raw = os.getenv("WALLET_LOOKBACK_BLOCKS")
WALLET_LOOKBACK_BLOCKS = int(_wallet_lookback_raw) if _wallet_lookback_raw else None
_wallet_events_raw = os.getenv("WALLET_MAX_EVENTS")
WALLET_MAX_EVENTS = int(_wallet_events_raw) if _wallet_events_raw else None


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #


class SignalRequest(BaseModel):
    price_weight: float = Field(0.65, ge=0.0, le=1.0)
    sentiment_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    wallet: Optional[str] = Field(
        default=None, description="Optional wallet identifier (not yet used)."
    )


class Recommendation(BaseModel):
    action: str
    probability: float
    rationale: Optional[str] = None


class ClusterInsight(BaseModel):
    id: int
    label: Optional[str] = None
    description: Optional[str] = None
    drivers: Optional[List[str]] = None
    metrics: Optional[Dict[str, float]] = None


class WalletSummaryPayload(BaseModel):
    wallet: str
    fetched_at: str
    holdings_text: str
    recent_activity_text: str
    eth_balance: float
    tokens: List[Dict[str, Any]] = Field(default_factory=list)
    staking_events_inferred: List[Dict[str, Any]] = Field(default_factory=list)
    tracked_tokens: List[Dict[str, Any]] = Field(default_factory=list)
    tracked_tokens_text: Optional[str] = None
    errors: List[str] = Field(default_factory=list)


class SignalResponse(BaseModel):
    recommendations: List[Recommendation]
    cluster: Optional[ClusterInsight] = None
    message: Optional[str] = None
    generated_at: str
    narrative: Optional[str] = None
    wallet_summary: Optional[WalletSummaryPayload] = None
    wallet_summary_error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Runtime helpers
# --------------------------------------------------------------------------- #


class RandomForestRuntime:
    """Wraps the pickled RF artifact + feature builder from StakingSignalTrainer."""

    def __init__(self, artifact_path: Path = RF_ARTIFACT_PATH) -> None:
        if not artifact_path.exists():
            raise FileNotFoundError(f"RandomForest artifact missing: {artifact_path}")

        self.trainer = StakingSignalTrainer()
        with artifact_path.open("rb") as fh:
            self.artifact = pickle.load(fh)

        self.model = self.artifact["model"]
        self.feature_names = self.artifact["feature_names"]
        self.trainer.numeric_imputer = self.artifact["numeric_imputer"]
        focus_data = self.artifact.get("focus")
        if isinstance(focus_data, dict):
            self.focus_from_training = FocusConfig(
                semantic_importance=focus_data.get("semantic_importance", 0.5),
                price_importance=focus_data.get("price_importance", 0.5),
            )
        elif isinstance(focus_data, FocusConfig):
            self.focus_from_training = focus_data
        else:
            self.focus_from_training = FocusConfig()

        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(self.artifact["label_classes"])

    def _latest_snapshot(self) -> pd.DataFrame:
        df = self.trainer._load_dataset()
        if df.empty:
            raise RuntimeError("Source dataset is empty.")
        return df.tail(1)

    def predict(
        self,
        *,
        price_weight: float,
        sentiment_weight: float,
    ) -> List[Recommendation]:
        focus = FocusConfig(
            semantic_importance=sentiment_weight,
            price_importance=price_weight,
        )
        latest_rows = self._latest_snapshot()
        features, _ = self.trainer._build_feature_frame(
            latest_rows, focus, training=False
        )

        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0.0
        extra_columns = [col for col in features.columns if col not in self.feature_names]
        if extra_columns:
            features = features.drop(columns=extra_columns)
        features = features[self.feature_names]

        proba = self.model.predict_proba(features)[0]
        ranked_indices = np.argsort(proba)[::-1][:3]

        bias_descriptor = "price action" if price_weight >= sentiment_weight else "semantic context"
        recommendations: List[Recommendation] = []
        for idx in ranked_indices:
            action = self.label_encoder.inverse_transform([idx])[0]
            rationale = f"Weighted {bias_descriptor} inputs favoured {action.replace('_', ' ')}."
            recommendations.append(
                Recommendation(action=action, probability=float(proba[idx]), rationale=rationale)
            )
        return recommendations


class ClusterRuntime:
    """Loads the clustering pickle + provides lightweight context."""

    CLUSTER_LABELS: Dict[int, Dict[str, Any]] = {
        0: {
            "label": "Restake skew",
            "description": "APR momentum trending higher alongside elevated withdrawer counts.",
            "drivers": [
                "Daily APR sits at the higher end of recent range.",
                "Withdrawals dominate flows but deposits remain elevated.",
            ],
        },
        1: {
            "label": "Baseline stake",
            "description": "Calmer flows with fewer withdrawers; APR cooling slightly.",
            "drivers": [
                "Withdrawal pressure muted relative to deposits.",
                "Sentiment leaning neutral; APR trend mostly flat.",
            ],
        },
        2: {
            "label": "Liquid bias",
            "description": "Liquidity preservation stance as netflows hover around zero.",
            "drivers": [
                "Withdrawer counts rising faster than deposits.",
                "Premium/discount stabilising near parity.",
            ],
        },
    }

    def __init__(self, artifact_path: Path = CLUSTER_ARTIFACT_PATH) -> None:
        if not artifact_path.exists():
            raise FileNotFoundError(f"Clustering artifact missing: {artifact_path}")
        with artifact_path.open("rb") as fh:
            self.artifacts: ClusterArtifacts = pickle.load(fh)

    def _latest_row(self) -> pd.Series:
        if ETHERFI_SOURCE.exists():
            df = pd.read_csv(ETHERFI_SOURCE)
        else:
            raise RuntimeError(f"EtherFi source not found: {ETHERFI_SOURCE}")
        if df.empty:
            raise RuntimeError("EtherFi dataset is empty.")
        return df.tail(1).squeeze()

    def classify(self) -> ClusterInsight:
        row = self._latest_row()
        feature_frame = pd.DataFrame([row], columns=row.index)
        for col in self.artifacts.feature_names:
            if col not in feature_frame:
                feature_frame[col] = 0.0
        feature_frame = feature_frame[self.artifacts.feature_names]

        imputed = self.artifacts.imputer.transform(feature_frame)
        scaled = self.artifacts.scaler.transform(imputed)
        cluster_id = int(self.artifacts.model.predict(scaled)[0])

        summary_row = {
            "daily_apr": row.get("daily_apr"),
            "withdraw": row.get("withdraw"),
            "deposit": row.get("deposit"),
            "daily_netflow": row.get("daily_netflow"),
            "withdrawers": row.get("withdrawers"),
            "depositors": row.get("depositors"),
        }

        metadata = self.CLUSTER_LABELS.get(cluster_id, {})
        return ClusterInsight(
            id=cluster_id,
            label=metadata.get("label"),
            description=metadata.get("description"),
            drivers=metadata.get("drivers"),
            metrics={
                key: float(value) if pd.notna(value) else np.nan
                for key, value in summary_row.items()
            },
        )


def _resolve_wallet_summary(wallet: Optional[str]) -> Tuple[Optional[WalletSummaryPayload], Optional[str]]:
    if wallet is None or wallet.strip() == "":
        return None, None
    if wallet_snapshot_fetch is None:
        return None, WALLET_IMPORT_ERROR or "wallet_snapshot module unavailable"
    try:
        summary = wallet_snapshot_fetch(
            wallet=wallet,
            lookback_blocks=WALLET_LOOKBACK_BLOCKS,
            max_events=WALLET_MAX_EVENTS,
        )
    except Exception as exc:  # pragma: no cover - network variance
        return None, str(exc)

    claude_bits = summary.get("summary_for_claude", {})
    payload = WalletSummaryPayload(
        wallet=summary.get("wallet", wallet),
        fetched_at=summary.get("fetched_at", datetime.now(timezone.utc).isoformat()),
        holdings_text=claude_bits.get("holdings_text", "No holdings text."),
        recent_activity_text=claude_bits.get(
            "recent_activity_text", "No recent transfer activity recorded."
        ),
        eth_balance=float(summary.get("eth_balance", 0.0)),
        tokens=summary.get("tokens", []),
        staking_events_inferred=summary.get("staking_events_inferred", []),
        tracked_tokens=summary.get("tracked_tokens", []),
        tracked_tokens_text=claude_bits.get("tracked_token_text"),
        errors=summary.get("errors", []),
    )
    return payload, None


def _compose_wallet_text(wallet_summary: WalletSummaryPayload | None) -> str:
    if wallet_summary is None:
        return "Wallet signal not provided."

    token_summaries = []
    for token in wallet_summary.tokens:
        balance = token.get("balance")
        if isinstance(balance, (int, float)) and balance > 0:
            token_summaries.append(f"{token.get('symbol', 'token')}: {balance:.4f}")
        if len(token_summaries) >= 4:
            break
    staking_note = ""
    if wallet_summary.staking_events_inferred:
        first = wallet_summary.staking_events_inferred[0]
        staking_note = (
            f"{first.get('direction', 'flow').title()} of "
            f"{first.get('value_human')} {first.get('token_symbol', '').upper()} "
            f"with {first.get('counterparty', 'unknown counterparty')}."
        )
    tracked_note = wallet_summary.tracked_tokens_text or ""
    return "\n".join(
        filter(
            None,
            [
                wallet_summary.holdings_text,
                f"Recent wallet activity: {wallet_summary.recent_activity_text}",
                f"Token balances: {', '.join(token_summaries) or 'n/a'}",
                f"Staking signal: {staking_note}" if staking_note else "",
                f"Tracked tokens: {tracked_note}" if tracked_note else "",
            ],
        )
    )


def _fallback_narrative(
    recommendations: List[Recommendation],
    cluster: ClusterInsight | None,
    price_weight: float,
    wallet_text: str,
) -> str:
    if recommendations:
        primary = recommendations[0]
        secondary = recommendations[1] if len(recommendations) > 1 else None
        actions_text = f"Top action leans {primary.action} ({primary.probability:.0%})."
        if secondary:
            actions_text += f" Backup: {secondary.action} ({secondary.probability:.0%})."
    else:
        actions_text = "Signal stack is still warming up."

    cluster_text = (
        f"Cluster {cluster.id}: {cluster.description}"
        if cluster
        else "Cluster context unavailable."
    )
    focus_text = (
        f"Bias is {price_weight:.0%} price vs {(1 - price_weight):.0%} sentiment."
    )
    return (
        f"{actions_text} {cluster_text} {focus_text} "
        f"Wallet context -> {wallet_text}"
    )


# --------------------------------------------------------------------------- #
# FastAPI wiring
# --------------------------------------------------------------------------- #


app = FastAPI(title="Model Signal Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
rf_runtime: Optional[RandomForestRuntime] = None
cluster_runtime: Optional[ClusterRuntime] = None
anthropic_client: Optional[Any] = None


def _build_narrative(
    recommendations: List[Recommendation],
    cluster: ClusterInsight | None,
    price_weight: float,
    wallet_summary: WalletSummaryPayload | None,
) -> Optional[str]:
    wallet_text = _compose_wallet_text(wallet_summary)
    if anthropic_client is None:
        return _fallback_narrative(recommendations, cluster, price_weight, wallet_text)

    top_actions = ", ".join(
        f"{rec.action} ({rec.probability:.0%})" for rec in recommendations[:3]
    )
    cluster_text = (
        f"Cluster {cluster.id}: {cluster.description}"
        if cluster
        else "Cluster data unavailable."
    )
    prompt = f"""
You are an Ethereum staking strategist that explains model outputs.

Action stack: {top_actions}
Cluster regime: {cluster_text}
Focus bias: {price_weight:.0%} price vs {1 - price_weight:.0%} semantic.
Wallet signal:
{wallet_text}

Write a paragraph for the UI summarizing what the user should know.
No markdown, no bullet lists, just plain text.
"""
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=150,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as exc:  
        print(f"[warn] Claude narrative failed: {exc}")
        return _fallback_narrative(
            recommendations, cluster, price_weight, wallet_text
        )


@app.on_event("startup")
def _initialise_runtimes() -> None:
    global rf_runtime, cluster_runtime, anthropic_client
    rf_runtime = RandomForestRuntime()
    cluster_runtime = ClusterRuntime()
    api_key = os.getenv("CLAUDE_API_KEY")
    if api_key and Anthropic is not None:
        anthropic_client = Anthropic(api_key=api_key)
        print("[claude] Anthropic client initialized for narratives.")
    else:
        anthropic_client = None
        print("[claude] Narrative generation disabled (missing key or package).")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "artifacts": {"rf": RF_ARTIFACT_PATH.exists(), "cluster": CLUSTER_ARTIFACT_PATH.exists()}}


@app.post("/signals", response_model=SignalResponse)
def signals(payload: SignalRequest) -> SignalResponse:
    if rf_runtime is None or cluster_runtime is None:
        raise HTTPException(status_code=503, detail="Model runtimes not ready.")

    sentiment_weight = (
        payload.sentiment_weight
        if payload.sentiment_weight is not None
        else 1.0 - payload.price_weight
    )
    sentiment_weight = np.clip(sentiment_weight, 0.0, 1.0)
    price_weight = np.clip(payload.price_weight, 0.0, 1.0)

    recommendations = rf_runtime.predict(
        price_weight=price_weight, sentiment_weight=sentiment_weight
    )
    cluster = cluster_runtime.classify()
    wallet_summary_payload, wallet_summary_error = _resolve_wallet_summary(payload.wallet)

    message = "Slider bias favours price precision." if price_weight >= sentiment_weight else "Slider bias leans semantic."
    if payload.wallet:
        if wallet_summary_payload:
            message += " Wallet inspector snapshot forwarded to Claude."
        elif wallet_summary_error:
            message += f" Wallet signal unavailable ({wallet_summary_error})."
        else:
            message += f" Wallet hint '{payload.wallet}' stored for future routing."

    narrative = _build_narrative(recommendations, cluster, price_weight, wallet_summary_payload)

    return SignalResponse(
        recommendations=recommendations,
        cluster=cluster,
        message=message,
        generated_at=datetime.now(timezone.utc).isoformat(),
        narrative=narrative,
        wallet_summary=wallet_summary_payload,
        wallet_summary_error=wallet_summary_error,
    )
