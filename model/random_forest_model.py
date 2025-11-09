"""
Stake / restake / liquid stake trainer backed by tree models.

This module consumes the merged ETH + news dataset stored in
    project-one/data/final_with_sentiment.csv

It fits two models:
    1. RandomForestClassifier (fast, robust baseline)
    2. XGBClassifier (optional, boosted tree alternative)

Both models predict one of three actions derived from `future_return`:

    - "stake"          → neutral view, (re)deploy capital normally
    - "restake"        → strong positive view, compound rewards
    - "liquid_stake"   → defensive posture, keep liquidity flexible

UI / API callers can pass a `FocusConfig` to rebalance semantic (FinBERT-based)
features versus raw market & staking features. This enables a UI slider that
leans more on "semantic context" vs "price precision" during both training
and inference.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Optional dependency – XGBoost is nice to have but not required.
try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None


# --------------------------------------------------------------------------- #
# Paths & column configuration
# --------------------------------------------------------------------------- #

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = WORKSPACE_ROOT / "data" / "final_with_sentiment.csv"

# Feature groups help us rebalance signals based on the UI slider.
SEMANTIC_BASE_COLUMNS = {
    "sent_pos",
    "sent_neg",
    "sent_neu",
    "has_news",
    "news_lag_hours",
}
SEMANTIC_PREFIXES = ("sentiment_label_",)

PRICE_BASE_COLUMNS = {
    "eth_price",
    "market_cap",
    "total_volume",
    "amount_staked",
    "earned_rewards",
    "validators",
    "marketshare",
    "days_since_deposit",
    "days_since_withdrawal",
    "hours_from_start",
    "day_of_week",
    "month",
    "is_weekend",
}

# Raw numeric columns expected in final_with_sentiment.csv
NUMERIC_SOURCE_COLUMNS = [
    "eth_price",
    "market_cap",
    "total_volume",
    "amount_staked",
    "earned_rewards",
    "validators",
    "marketshare",
    "days_since_deposit",
    "days_since_withdrawal",
    "sent_pos",
    "sent_neg",
    "sent_neu",
    "has_news",
]

# Time columns we know how to parse
DATE_COLUMNS = ["timestamp", "published_at"]

# Categorical columns we one-hot encode
CATEGORICAL_COLUMNS = ["sentiment_label", "topic"]


# --------------------------------------------------------------------------- #
# Config dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class FocusConfig:
    """
    Represents how much weight to give semantic vs price features.

    semantic_importance, price_importance ∈ [0, 1].

    Internally we map value → multiplier in the 0.5 → 1.5 range so the slider
    has a tangible effect without blowing up feature scales:
        0.0 → 0.5x
        0.5 → 1.0x
        1.0 → 1.5x
    """

    semantic_importance: float = 0.5
    price_importance: float = 0.5

    @staticmethod
    def _scale(value: float) -> float:
        value = min(max(float(value), 0.0), 1.0)
        return 0.5 + value  # 0 → 0.5, 1 → 1.5

    def multipliers(self) -> Tuple[float, float]:
        """Return (semantic_multiplier, price_multiplier)."""
        return tuple(self._scale(v) for v in (self.semantic_importance, self.price_importance))

    @classmethod
    def from_slider(cls, semantic_focus: float) -> "FocusConfig":
        """
        Helper for UI slider.

        User chooses semantic_focus ∈ [0, 1]; price focus is implied as 1 - semantic_focus.
        """
        return cls(semantic_focus, 1.0 - semantic_focus)


@dataclass
class ActionThresholds:
    """
    Heuristics that transform `future_return` into a discrete action.

    restake_min      : minimum positive future_return to justify restaking
    liquid_stake_max : maximum negative future_return to trigger defensive mode
    """

    restake_min: float = 0.02        # >= +2% return → restake
    liquid_stake_max: float = -0.02  # <= -2% return → liquid_stake

    def __post_init__(self) -> None:
        if self.restake_min <= self.liquid_stake_max:
            raise ValueError("restake_min must be greater than liquid_stake_max")

    def classify(self, future_return: float | None) -> str:
        """Map numeric future_return into one of: stake / restake / liquid_stake."""
        if future_return is None or pd.isna(future_return):
            return "stake"
        if future_return >= self.restake_min:
            return "restake"
        if future_return <= self.liquid_stake_max:
            return "liquid_stake"
        return "stake"


@dataclass
class ModelResult:
    model_type: Literal["random_forest", "xgboost"]
    model: Any
    metrics: Dict[str, Any]
    focus: FocusConfig
    feature_names: List[str]


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

class StakingSignalTrainer:
    """
    End-to-end helper that loads data, engineers features, and trains models.

    Usage pattern:

        trainer = StakingSignalTrainer()
        focus = FocusConfig.from_slider(semantic_focus=0.7)
        result = trainer.train("random_forest", focus=focus)

        latest = trainer._load_dataset().tail(1)
        recs = trainer.recommend_actions(latest)
    """

    def __init__(
        self,
        data_path: Path = DEFAULT_DATA_PATH,
        thresholds: ActionThresholds | None = None,
        random_state: int = 42,
    ) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        self.thresholds = thresholds or ActionThresholds()
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        # Stores trained models + metadata keyed by model_type
        self.artifacts: Dict[str, Dict[str, Any]] = {}
        self.numeric_imputer: SimpleImputer | None = None

    # --------------------- dataset + feature engineering -------------------- #

    def _load_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)

        # Sanity check: numeric columns required for features
        missing_cols = [col for col in NUMERIC_SOURCE_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset missing numeric columns: {missing_cols}")

        if "future_return" not in df.columns:
            raise ValueError("Dataset must contain a 'future_return' column for labeling.")

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()

        # Parse timestamps
        for col in DATE_COLUMNS:
            if col in frame.columns:
                frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")

        frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
        ts = frame["timestamp"]

        # Basic time features
        frame["hours_from_start"] = (ts - ts.min()).dt.total_seconds().div(3600.0)
        frame["day_of_week"] = ts.dt.dayofweek
        frame["month"] = ts.dt.month
        frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)

        # How "old" the attached news is in hours
        if "published_at" in frame.columns:
            lag = (frame["timestamp"] - frame["published_at"]).dt.total_seconds().div(3600.0)
            if lag.isna().all():
                frame["news_lag_hours"] = 0.0
            else:
                frame["news_lag_hours"] = lag.fillna(lag.median())
        else:
            frame["news_lag_hours"] = 0.0

        return frame

    def _encode_actions(self, df: pd.DataFrame) -> np.ndarray:
        """
        Turn future_return into a discrete label, then into integer classes.
        """
        actions = df["future_return"].apply(self.thresholds.classify)
        return self.label_encoder.fit_transform(actions)

    def _prepare_numeric(self, df: pd.DataFrame, *, training: bool) -> pd.DataFrame:
        numeric_cols = list(
            dict.fromkeys(
                NUMERIC_SOURCE_COLUMNS
                + ["hours_from_start", "day_of_week", "month", "is_weekend", "news_lag_hours"]
            )
        )
        present_cols = [col for col in numeric_cols if col in df.columns]
        num_df = df[present_cols].apply(pd.to_numeric, errors="coerce")
        num_df = num_df.replace([np.inf, -np.inf], np.nan)
        if training:
            self.numeric_imputer = SimpleImputer(strategy="median")
            imputed = self.numeric_imputer.fit_transform(num_df)
        else:
            if self.numeric_imputer is None:
                raise RuntimeError("Numeric imputer has not been fitted; call train() before inference.")
            imputed = self.numeric_imputer.transform(num_df)
        num_df = pd.DataFrame(imputed, columns=num_df.columns, index=num_df.index)
        return num_df

    def _prepare_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        present = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
        if not present:
            return pd.DataFrame(index=df.index)
        cat_df = pd.get_dummies(df[present].fillna("unknown"), prefix=present, dtype=int)
        return cat_df

    def _apply_focus_weights(self, features: pd.DataFrame, focus: FocusConfig) -> pd.DataFrame:
        """
        Rescale semantic vs price-related features based on the focus config.
        """
        semantic_weight, price_weight = focus.multipliers()
        weighted = features.copy()

        for col in weighted.columns:
            if col in SEMANTIC_BASE_COLUMNS or col.startswith(SEMANTIC_PREFIXES):
                weighted[col] *= semantic_weight
            elif col in PRICE_BASE_COLUMNS:
                weighted[col] *= price_weight

        return weighted

    def _build_feature_frame(
        self,
        df: pd.DataFrame,
        focus: FocusConfig,
        *,
        training: bool,
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Build a feature matrix (and labels if training=True) from raw dataframe.
        """
        working = self._add_temporal_features(df)

        if training:
            # Drop rows without future_return for training
            working = working.dropna(subset=["future_return"]).reset_index(drop=True)
            labels = self._encode_actions(working)
        else:
            labels = None

        numeric = self._prepare_numeric(working, training=training)
        categoricals = self._prepare_categoricals(working)

        features = pd.concat([numeric, categoricals], axis=1)
        features = features.reindex(sorted(features.columns), axis=1)
        features = self._apply_focus_weights(features, focus)

        return features, labels

    # -------------------------- training / evaluation ----------------------- #

    def _initialise_model(
        self,
        model_type: Literal["random_forest", "xgboost"],
        num_classes: int,
    ) -> Any:
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=3,
                min_samples_split=4,
                class_weight="balanced_subsample",
                random_state=self.random_state,
                n_jobs=-1,
            )

        if model_type == "xgboost":
            if XGBClassifier is None:
                raise ImportError("xgboost is not installed. Run `pip install xgboost`.")
            return XGBClassifier(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.75,
                objective="multi:softprob",
                num_class=num_classes,
                eval_metric="mlogloss",
                reg_lambda=1.0,
                reg_alpha=0.1,
                random_state=self.random_state,
                tree_method="hist",
                verbosity=0,
            )

        raise ValueError(f"Unsupported model_type: {model_type}")

    def _evaluate(self, model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        y_pred = model.predict(X_test)

        report = classification_report(
            y_test,
            y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_test, y_pred).tolist()
        predicted_counts = dict(
            zip(
                self.label_encoder.classes_,
                np.bincount(y_pred, minlength=len(self.label_encoder.classes_)),
            )
        )

        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "classification_report": report,
            "confusion_matrix": matrix,
            "predicted_distribution": predicted_counts,
        }

    def train(
        self,
        model_type: Literal["random_forest", "xgboost"] = "random_forest",
        *,
        focus: FocusConfig | None = None,
        test_size: float = 0.2,
    ) -> ModelResult:
        """
        Train a model and capture evaluation metadata.

        Parameters
        ----------
        model_type : "random_forest" or "xgboost"
        focus      : FocusConfig giving semantic vs price weighting
        test_size  : fraction of the dataset to reserve for validation
        """
        focus = focus or FocusConfig()

        df = self._load_dataset()
        features, labels = self._build_feature_frame(df, focus, training=True)
        if labels is None:
            raise RuntimeError("Training requires labels; 'future_return' column missing or empty.")

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=self.random_state,
            stratify=labels,
        )

        model = self._initialise_model(model_type, num_classes=len(self.label_encoder.classes_))
        model.fit(X_train, y_train)
        metrics = self._evaluate(model, X_test, y_test)

        focus_snapshot = asdict(focus)
        artifact = {
            "model_type": model_type,
            "model": model,
            "focus": focus,
            "focus_snapshot": focus_snapshot,
            "feature_names": list(features.columns),
            "imputer": self.numeric_imputer,
            "label_classes": self.label_encoder.classes_.tolist(),
        }
        self.artifacts[model_type] = artifact

        return ModelResult(
            model_type=model_type,
            model=model,
            metrics=metrics,
            focus=focus,
            feature_names=list(features.columns),
        )

    # ---------------------------- inference helpers ------------------------- #

    def _ensure_model_ready(self, model_type: str) -> Dict[str, Any]:
        if model_type not in self.artifacts:
            raise ValueError(f"Model '{model_type}' has not been trained yet.")
        artifact = self.artifacts[model_type]
        if self.numeric_imputer is None and artifact.get("imputer") is not None:
            self.numeric_imputer = artifact["imputer"]
        return artifact

    def export_model(
        self,
        model_type: Literal["random_forest", "xgboost"] = "random_forest",
        output_path: str | Path = "artifacts/random_forest_model.pkl",
    ) -> Path:
        """
        Persist a trained model artifact (model, imputer, label classes, focus).
        """
        artifact = self._ensure_model_ready(model_type)
        focus_data = artifact.get("focus_snapshot")
        if focus_data is None and isinstance(artifact.get("focus"), FocusConfig):
            focus_data = asdict(artifact["focus"])
        bundle = {
            "model_type": model_type,
            "model": artifact["model"],
            "focus": focus_data,
            "feature_names": artifact["feature_names"],
            "numeric_imputer": artifact["imputer"],
            "label_classes": artifact["label_classes"],
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as fh:
            pickle.dump(bundle, fh)
        return output_path

    def recommend_actions(
        self,
        latest_rows: pd.DataFrame,
        *,
        model_type: Literal["random_forest", "xgboost"] = "random_forest",
        focus_override: FocusConfig | None = None,
        top_k: int = 3,
    ) -> List[List[Dict[str, float | str]]]:
        """
        Return ranked action recommendations for each input row.

        Output shape:
            [
              [ {"action": "restake", "probability": 0.62},
                {"action": "stake",   "probability": 0.28},
                {"action": "liquid_stake", "probability": 0.10} ],
              ...
            ]
        """
        artifact = self._ensure_model_ready(model_type)
        focus = focus_override or artifact["focus"]
        features, _ = self._build_feature_frame(latest_rows, focus, training=False)

        # Align feature columns with what the model saw during training.
        feature_names = artifact["feature_names"]
        for col in feature_names:
            if col not in features.columns:
                features[col] = 0.0
        extra_cols = [col for col in features.columns if col not in feature_names]
        if extra_cols:
            features = features.drop(columns=extra_cols)
        features = features[feature_names]

        model = artifact["model"]
        proba = model.predict_proba(features)
        decoded: List[List[Dict[str, float | str]]] = []

        for row in proba:
            ranked_indices = np.argsort(row)[::-1][:top_k]
            decoded.append(
                [
                    {
                        "action": self.label_encoder.inverse_transform([idx])[0],
                        "probability": float(row[idx]),
                    }
                    for idx in ranked_indices
                ]
            )

        return decoded


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #

def _pretty_print_metrics(model_name: str, metrics: Dict[str, Any]) -> None:
    print(f"\n📊 {model_name} metrics")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print("Predicted class distribution:")
    for action, count in metrics["predicted_distribution"].items():
        print(f"  - {action:<14} {count}")
    print("Classification report:")
    print(json.dumps(metrics["classification_report"], indent=2))


if __name__ == "__main__":
    # Example: slightly semantic-leaning focus
    trainer = StakingSignalTrainer()
    ui_focus = FocusConfig.from_slider(semantic_focus=0.65)  # 65% semantic, 35% price

    # Train both RF and XGBoost (if installed)
    for name in ("random_forest", "xgboost"):
        try:
            result = trainer.train(model_type=name, focus=ui_focus)
        except ImportError as exc:
            print(f"[warn] Skipping {name}: {exc}")
            continue

        _pretty_print_metrics(name, result.metrics)
        export_path = WORKSPACE_ROOT / "artifacts" / f"{name}_model.pkl"
        saved_path = trainer.export_model(name, export_path)
        print(f"Saved {name} artifact to {saved_path}")

    # Example: score the latest row with whichever model ran last.
    if trainer.artifacts:
        latest_snapshot = trainer._load_dataset().tail(1)
        final_model_type = list(trainer.artifacts.keys())[-1]
        scores = trainer.recommend_actions(latest_snapshot, model_type=final_model_type)
        print(f"\nLatest recommendation using {final_model_type}:")
        for rank, candidate in enumerate(scores[0], start=1):
            print(f"  {rank}. {candidate['action']} ({candidate['probability']:.2%})")
