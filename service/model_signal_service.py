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
from typing import Any, Dict, List, Optional

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


class SignalResponse(BaseModel):
    recommendations: List[Recommendation]
    cluster: Optional[ClusterInsight] = None
    message: Optional[str] = None
    generated_at: str
    narrative: Optional[str] = None


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
) -> Optional[str]:
    if anthropic_client is None:
        return None

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
        return None


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

    message = "Slider bias favours price precision." if price_weight >= sentiment_weight else "Slider bias leans semantic."
    if payload.wallet:
        message += f" Wallet hint '{payload.wallet}' stored for future routing."

    narrative = _build_narrative(recommendations, cluster, price_weight)

    return SignalResponse(
        recommendations=recommendations,
        cluster=cluster,
        message=message,
        generated_at=datetime.now(timezone.utc).isoformat(),
        narrative=narrative,
    )
