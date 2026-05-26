"""ML filter for trade signals.

Purpose: given features at signal time, predict the probability that the trade
would have closed in profit. The live loop only acts on a signal if the model's
proba clears ``ml.min_proba_to_trade``. With <``min_train_samples`` closed
trades, the filter is a no-op (returns 1.0 = always accept) so the bot can
bootstrap.

Self-improvement comes from periodic retraining: every ``retrain_every_n_trades``
closed trades, we rebuild the model from the journal. We use a gradient-boosted
decision tree (HistGradientBoostingClassifier) because it handles mixed-scale
features without manual scaling and is fast on CPU.

Important caveats (worth being honest about):
  * The journal only contains *features at signals we acted on*. The model is
    therefore biased toward setups the strategy already likes. That's fine as a
    *filter* but it cannot discover new setups.
  * With <500 trades the model is mostly noise. Don't size up based on it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .indicators import FEATURE_COLUMNS
from .logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class TrainReport:
    n_samples: int
    n_features: int
    cv_auc: float
    cv_acc: float
    cv_logloss: float
    class_balance: Dict[int, int]


class MLFilter:
    def __init__(self, ml_cfg: Dict[str, Any]):
        self.cfg = ml_cfg
        self.enabled: bool = bool(ml_cfg.get("enabled", True))
        self.model_path = Path(ml_cfg["model_path"])
        self.min_train_samples = int(ml_cfg.get("min_train_samples", 100))
        self.min_proba = float(ml_cfg.get("min_proba_to_trade", 0.55))
        self.retrain_every = int(ml_cfg.get("retrain_every_n_trades", 25))
        self.feature_columns: List[str] = list(FEATURE_COLUMNS)
        self.model: Optional[HistGradientBoostingClassifier] = None
        self._last_trained_at_count: int = 0
        self._load_if_exists()

    # ------------------------------------------------------------------ persistence
    def _load_if_exists(self) -> None:
        if self.model_path.exists():
            try:
                payload = joblib.load(self.model_path)
                self.model = payload["model"]
                # Use the schema the model was trained on -- not the current
                # FEATURE_COLUMNS -- so predict_proba feeds the right shape.
                # If the bot's feature set has since grown, ``maybe_retrain``
                # will rebuild on the new schema once enough trades accrue.
                self.feature_columns = list(payload.get("features", FEATURE_COLUMNS))
                self._last_trained_at_count = int(payload.get("trained_at_count", 0))
                # Detect schema drift so the live loop can flag it.
                self._schema_drift = (set(self.feature_columns) != set(FEATURE_COLUMNS))
                if self._schema_drift:
                    log.warning(
                        "ML model trained on a different feature schema "
                        "(model has %d cols, current build has %d). "
                        "Predictions still work; will retrain on next cycle.",
                        len(self.feature_columns), len(FEATURE_COLUMNS),
                    )
                log.info(
                    "ML model loaded from %s (trained at %d samples)",
                    self.model_path, self._last_trained_at_count,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("Failed to load ML model: %s. Will retrain when ready.", e)
                self.model = None
                self._schema_drift = False
        else:
            self._schema_drift = False

    def _save(self, report: TrainReport) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "features": self.feature_columns,
                "trained_at_count": self._last_trained_at_count,
                "report": report.__dict__,
            },
            self.model_path,
        )
        log.info("ML model saved to %s", self.model_path)

    # ------------------------------------------------------------------ inference
    def predict_proba_win(self, features: Dict[str, float]) -> float:
        """Probability the candidate trade is a win. Returns 1.0 if no model yet."""
        if not self.enabled or self.model is None:
            return 1.0
        x = self._row_from_features(features)
        if x is None:
            return 1.0  # missing features: don't block
        try:
            proba = self.model.predict_proba(x)[0, 1]
            return float(proba)
        except Exception as e:  # noqa: BLE001
            log.warning("ML predict failed (%s); accepting signal", e)
            return 1.0

    def should_trade(self, proba: float) -> bool:
        if not self.enabled or self.model is None:
            return True
        return proba >= self.min_proba

    def status(self) -> Dict[str, Any]:
        """Expose current model state for the dashboard. Reads the saved
        report (if any) from the model file so the live loop doesn't need to
        keep it in memory."""
        out: Dict[str, Any] = {
            "enabled": self.enabled,
            "loaded": self.model is not None,
            "min_proba": self.min_proba,
            "min_train_samples": self.min_train_samples,
            "retrain_every": self.retrain_every,
            "trained_at_count": self._last_trained_at_count,
            "feature_columns": list(self.feature_columns),
            "n_features_current": len(FEATURE_COLUMNS),
            "schema_drift": bool(self._schema_drift),
            "model_path": str(self.model_path),
            "report": None,
        }
        if self.model_path.exists():
            try:
                payload = joblib.load(self.model_path)
                rep = payload.get("report")
                if isinstance(rep, dict):
                    out["report"] = rep
            except Exception:  # noqa: BLE001
                pass
        return out

    # ------------------------------------------------------------------ training
    def maybe_retrain(self, journal, symbol: Optional[str] = None) -> Optional[TrainReport]:
        if not self.enabled:
            return None
        n_closed = journal.closed_count(symbol=symbol)
        if n_closed < self.min_train_samples:
            return None
        # Force retrain if the schema drifted (we added/removed features) so
        # the live model uses the latest feature set as soon as we have data.
        if self._schema_drift and n_closed >= self.min_train_samples:
            log.info("ML schema drift detected -- forcing retrain on new features")
        elif n_closed - self._last_trained_at_count < self.retrain_every and self.model is not None:
            return None
        # On retrain, snap to the live FEATURE_COLUMNS list so the model
        # learns the latest features.
        self.feature_columns = list(FEATURE_COLUMNS)
        self._schema_drift = False
        return self.retrain(journal, symbol=symbol)

    def retrain(self, journal, symbol: Optional[str] = None) -> Optional[TrainReport]:
        df = journal.closed_trades_df(symbol=symbol)
        if df.empty:
            return None

        X, y = self._build_xy(df)
        if X is None or len(X) < self.min_train_samples:
            log.info(
                "Not enough usable samples to train ML filter (have %d, need %d)",
                0 if X is None else len(X), self.min_train_samples,
            )
            return None

        # Class balance check: if we have <5 of a class, skip training
        unique, counts = np.unique(y, return_counts=True)
        balance = {int(u): int(c) for u, c in zip(unique, counts)}
        if len(unique) < 2 or min(counts) < 5:
            log.info("Skipping train: class balance too skewed (%s)", balance)
            return None

        # Time-series CV for honest estimates
        n_splits = min(5, max(2, len(X) // 50))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        aucs, accs, lls = [], [], []
        for tr_idx, va_idx in tscv.split(X):
            mdl = self._make_model(n_samples=len(tr_idx))
            mdl.fit(X[tr_idx], y[tr_idx])
            p = mdl.predict_proba(X[va_idx])[:, 1]
            yhat = (p >= 0.5).astype(int)
            try:
                aucs.append(roc_auc_score(y[va_idx], p))
            except ValueError:
                pass
            accs.append(accuracy_score(y[va_idx], yhat))
            lls.append(log_loss(y[va_idx], np.clip(p, 1e-6, 1 - 1e-6)))

        # Final fit on all data
        self.model = self._make_model(n_samples=len(X))
        self.model.fit(X, y)
        self._last_trained_at_count = len(X)

        report = TrainReport(
            n_samples=len(X),
            n_features=X.shape[1],
            cv_auc=float(np.mean(aucs)) if aucs else float("nan"),
            cv_acc=float(np.mean(accs)) if accs else float("nan"),
            cv_logloss=float(np.mean(lls)) if lls else float("nan"),
            class_balance=balance,
        )
        log.info(
            "ML retrained: n=%d feat=%d cv_auc=%.3f cv_acc=%.3f cv_ll=%.3f balance=%s",
            report.n_samples, report.n_features, report.cv_auc, report.cv_acc,
            report.cv_logloss, report.class_balance,
        )
        self._save(report)
        return report

    # ------------------------------------------------------------------ internals
    def _make_model(self, n_samples: int = 0) -> HistGradientBoostingClassifier:
        # Early stopping helps a lot on big datasets but starves the model on
        # small ones (the held-out validation slice becomes pure noise).
        # Below ~1000 samples we let it train to completion with stronger
        # regularisation instead.
        use_es = n_samples >= 1000
        return HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=300 if use_es else 200,
            l2_regularization=1.0 if use_es else 2.0,
            early_stopping=use_es,
            validation_fraction=0.15,
            random_state=42,
        )

    def _build_xy(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        rows: List[List[float]] = []
        ys: List[int] = []
        for _, r in df.iterrows():
            feats_json = r.get("features_json")
            if not feats_json:
                continue
            try:
                feats = json.loads(feats_json)
            except Exception:  # noqa: BLE001
                continue
            row = [feats.get(c, np.nan) for c in self.feature_columns]
            # Side-aware: encode side as a feature so the same model handles long+short
            row.append(1.0 if r["side"] == "buy" else 0.0)
            rows.append(row)
            ys.append(int(r["outcome"]))
        if not rows:
            return None, None
        X = np.array(rows, dtype=float)
        y = np.array(ys, dtype=int)
        # HistGradientBoostingClassifier handles NaN natively
        return X, y

    def _row_from_features(self, features: Dict[str, float]) -> Optional[np.ndarray]:
        if not features:
            return None
        row = [features.get(c, np.nan) for c in self.feature_columns]
        # Side feature is appended at predict time by the caller via features["__side_buy"]
        row.append(float(features.get("__side_buy", 0.0)))
        return np.array([row], dtype=float)
