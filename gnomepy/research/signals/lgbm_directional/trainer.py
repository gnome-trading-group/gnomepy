"""
Walk-forward training, hyperparameter tuning, and label creation for the
LightGBM directional prediction signal.
"""

import datetime
import itertools
import random
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from gnomepy.data.cached_client import CachedMarketDataClient
from gnomepy.data.types import SchemaType
from gnomepy.registry.api import RegistryClient

from .features import FEATURE_NAMES, extract_features_bulk
from .registry import ModelRegistry

# -----------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_idx: int
    model: lgb.Booster
    metrics: dict = field(default_factory=dict)


@dataclass
class TuningResult:
    best_params: dict
    all_results: list[dict] = field(default_factory=list)


# -----------------------------------------------------------------------
# Default param grid
# -----------------------------------------------------------------------

DEFAULT_PARAM_GRID: dict[str, list] = {
    "num_leaves": [15, 31, 63],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_samples": [50, 100, 200],
    "feature_fraction": [0.6, 0.8, 1.0],
}

# Fixed defaults (not tuned)
FIXED_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
}


class LGBMTrainer:
    """Train, tune, and evaluate LightGBM directional models."""

    def __init__(
        self,
        listing_id: int,
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
        schema_type: SchemaType = SchemaType.MBP_10,
        horizon: int = 20,
        threshold_bps: float = 0.0,
        market_data_client: CachedMarketDataClient | None = None,
        registry_client: RegistryClient | None = None,
    ):
        self.listing_id = listing_id
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.schema_type = schema_type
        self.horizon = horizon
        self.threshold_bps = threshold_bps

        self.market_data_client = market_data_client or CachedMarketDataClient(
            bucket="gnome-market-data-archive-dev",
            aws_profile_name="AWSAdministratorAccess-443370708724",
        )
        self.registry_client = registry_client or RegistryClient(
            api_key="9WPV7CfeqXa578yVYlxdG3kCPFzACr7YaMU0UVma",
        )

        result = self.registry_client.get_listing(listing_id=listing_id)
        if not result:
            raise ValueError(f"Unable to find listing_id: {listing_id}")
        self.listing = result[0]

        # Lazily loaded data
        self._raw_df: pd.DataFrame | None = None
        self._features: pd.DataFrame | None = None
        self._labels: pd.Series | None = None
        # Best params from tuning (used by train() if params=None)
        self._best_params: dict | None = None

    # ------------------------------------------------------------------
    # Data loading & preparation
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Fetch MBP10 data and convert to DataFrame."""
        if self._raw_df is not None:
            return self._raw_df

        print(f"Loading data for listing {self.listing_id} ...")
        data_store = self.market_data_client.get_data(
            exchange_id=self.listing.exchange_id,
            security_id=self.listing.security_id,
            start_datetime=self.start_datetime,
            end_datetime=self.end_datetime,
            schema_type=self.schema_type,
        )
        df = data_store.to_df(price_type="float", size_type="float")

        if "midPrice" not in df.columns:
            df["midPrice"] = (df["bidPrice0"] + df["askPrice0"]) / 2.0

        df = df.dropna(subset=["bidPrice0", "askPrice0", "midPrice"])

        if "ts_event" in df.columns:
            df = df.sort_values("ts_event")
        elif "timestamp" in df.columns:
            df = df.sort_values("timestamp")

        df = df.reset_index(drop=True)
        self._raw_df = df
        print(f"  Loaded {len(df)} rows")
        return df

    def prepare(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load data, compute features, create labels. Returns (X, y)."""
        df = self.load_data()
        X = self._compute_features(df)
        y = self._create_labels(df)

        # Align: drop rows where either features or labels are NaN
        valid = X.notna().all(axis=1) & y.notna()
        X = X.loc[valid].reset_index(drop=True)
        y = y.loc[valid].reset_index(drop=True)

        self._features = X
        self._labels = y

        up_pct = (y == 1.0).mean()
        down_pct = (y == 0.0).mean()
        neutral_pct = (y == 0.5).mean()
        print(f"  {len(X)} usable samples, class split: up={up_pct:.3f} neutral={neutral_pct:.3f} down={down_pct:.3f}")
        return X, y

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._features is not None:
            return self._features
        return extract_features_bulk(df)

    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Soft labels based on future mid-price movement.

        Label = 1.0 if ret > threshold
        Label = 0.0 if ret < -threshold
        Label = 0.5 for neutral (within dead-zone)
        NaN   for insufficient future data (last `horizon` rows)
        """
        if self._labels is not None:
            return self._labels

        mid = df["midPrice"].values
        n = len(mid)
        labels = pd.Series(np.nan, index=df.index, dtype="float64")

        future_mid = np.empty(n)
        future_mid[:] = np.nan
        if n > self.horizon:
            future_mid[: n - self.horizon] = mid[self.horizon:]

        ret = (future_mid - mid) / mid  # fractional return
        threshold = self.threshold_bps / 1e4

        labels[ret > threshold] = 1.0
        labels[ret < -threshold] = 0.0

        # Neutral zone: return exists but within threshold
        has_future = ~np.isnan(future_mid)
        in_deadzone = has_future & (np.abs(ret) <= threshold)
        labels[in_deadzone] = 0.5

        return labels

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    def walk_forward_validate(
        self,
        params: dict,
        train_window: int,
        val_window: int,
        step_size: int | None = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
    ) -> list[FoldResult]:
        """Slide a training/validation window through the data temporally.

        Parameters
        ----------
        params : dict
            LightGBM params (merged with FIXED_PARAMS).
        train_window : int
            Number of rows in each training split.
        val_window : int
            Number of rows in each validation split.
        step_size : int, optional
            How far to slide between folds. Defaults to ``val_window``.
        num_boost_round : int
        early_stopping_rounds : int

        Returns
        -------
        list[FoldResult]
        """
        X, y = self.prepare()
        n = len(X)
        step_size = step_size or val_window
        merged_params = {**FIXED_PARAMS, **params}

        fold_results: list[FoldResult] = []
        fold_idx = 0
        start = 0

        while start + train_window + val_window <= n:
            train_end = start + train_window
            val_end = train_end + val_window

            X_train = X.iloc[start:train_end]
            y_train = y.iloc[start:train_end]
            X_val = X.iloc[train_end:val_end]
            y_val = y.iloc[train_end:val_end]

            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
            model = lgb.train(
                merged_params,
                dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dval],
                callbacks=callbacks,
            )

            probs = model.predict(X_val)

            # Evaluate classification metrics on hard-labeled samples only
            hard_mask = (y_val == 0.0) | (y_val == 1.0)
            if hard_mask.sum() > 0:
                y_hard = y_val[hard_mask].values.astype(int)
                probs_hard = probs[hard_mask.values]
                preds_hard = (probs_hard > 0.5).astype(int)
                metrics = {
                    "auc": float(roc_auc_score(y_hard, probs_hard)),
                    "accuracy": float(accuracy_score(y_hard, preds_hard)),
                    "precision": float(precision_score(y_hard, preds_hard, zero_division=0)),
                    "recall": float(recall_score(y_hard, preds_hard, zero_division=0)),
                    "f1": float(f1_score(y_hard, preds_hard, zero_division=0)),
                    "log_loss": float(log_loss(y_hard, probs_hard)),
                }
            else:
                metrics = {}

            # Neutral calibration: how close does model predict to 0.5 for neutral samples
            neutral_mask = y_val == 0.5
            if neutral_mask.sum() > 0:
                metrics["neutral_mae"] = float(np.abs(probs[neutral_mask.values] - 0.5).mean())
                metrics["neutral_pct"] = float(neutral_mask.mean())

            fold_results.append(FoldResult(fold_idx=fold_idx, model=model, metrics=metrics))
            fold_idx += 1
            start += step_size

        return fold_results

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------

    def tune_hyperparameters(
        self,
        param_grid: dict[str, list] | None = None,
        train_window: int = 50000,
        val_window: int = 10000,
        n_random: int | None = None,
        metric: str = "auc",
        num_boost_round: int = 500,
    ) -> TuningResult:
        """Grid/random search over param combinations via walk-forward CV.

        Parameters
        ----------
        param_grid : dict
            Maps param name -> list of candidate values.
        train_window, val_window : int
            Passed through to ``walk_forward_validate``.
        n_random : int, optional
            If set, sample ``n_random`` combinations randomly instead of
            running the full grid.
        metric : str
            OOS metric to optimise (key in ``FoldResult.metrics``).
        num_boost_round : int

        Returns
        -------
        TuningResult
        """
        if param_grid is None:
            param_grid = DEFAULT_PARAM_GRID

        keys = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in keys]))

        if n_random is not None and n_random < len(combos):
            combos = random.sample(combos, n_random)

        print(f"Tuning: {len(combos)} param combos, metric={metric}")

        all_results: list[dict] = []

        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            folds = self.walk_forward_validate(
                params, train_window, val_window, num_boost_round=num_boost_round,
            )
            mean_metric = float(np.mean([f.metrics[metric] for f in folds]))
            entry = {"params": params, f"mean_{metric}": mean_metric, "n_folds": len(folds)}
            all_results.append(entry)
            print(f"  [{i+1}/{len(combos)}] mean_{metric}={mean_metric:.4f}  {params}")

        # Sort descending (higher metric = better for auc/accuracy/f1)
        all_results.sort(key=lambda r: r[f"mean_{metric}"], reverse=True)
        best = all_results[0]["params"]
        self._best_params = best

        print(f"\nBest params: {best}  (mean_{metric}={all_results[0][f'mean_{metric}']:.4f})")
        return TuningResult(best_params=best, all_results=all_results)

    # ------------------------------------------------------------------
    # Final training
    # ------------------------------------------------------------------

    def train(
        self,
        params: dict | None = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        register: bool = True,
        registry: ModelRegistry | None = None,
    ) -> lgb.Booster:
        """Train on all data with a held-out tail for early stopping.

        Parameters
        ----------
        params : dict, optional
            LightGBM params. If ``None``, uses best from
            ``tune_hyperparameters`` (must have been called first).
        num_boost_round : int
        early_stopping_rounds : int
        register : bool
            If ``True``, save to the model registry.
        registry : ModelRegistry, optional
            Registry instance to use. If ``None`` and ``register=True``,
            a default is created at ``./models``.

        Returns
        -------
        lgb.Booster
        """
        if params is None:
            if self._best_params is None:
                raise ValueError(
                    "No params provided and tune_hyperparameters has not been called."
                )
            params = self._best_params

        X, y = self.prepare()
        merged_params = {**FIXED_PARAMS, **params}

        # Hold out last 15% for early stopping
        split = int(len(X) * 0.85)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_val, y_val = X.iloc[split:], y.iloc[split:]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        model = lgb.train(
            merged_params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        # Run walk-forward to get OOS metrics for metadata
        fold_results = self.walk_forward_validate(
            params,
            train_window=int(len(X) * 0.6),
            val_window=int(len(X) * 0.15),
        )

        if register:
            if registry is None:
                registry = ModelRegistry(base_dir="./models")
            metadata = self._build_metadata(params, model, fold_results)
            version = registry.register(self.listing_id, model, metadata)
            print(f"Registered as {version}")

        return model

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, fold_results: list[FoldResult]) -> dict:
        """Print and return summary of walk-forward results."""
        all_metrics: dict[str, list[float]] = {}
        print("\n=== Walk-Forward Report ===\n")

        for fr in fold_results:
            print(f"Fold {fr.fold_idx}: {fr.metrics}")
            for k, v in fr.metrics.items():
                all_metrics.setdefault(k, []).append(v)

        avg = {k: float(np.mean(v)) for k, v in all_metrics.items()}
        print(f"\nAverage: {avg}")

        # Feature importance from last fold
        if fold_results:
            last_model = fold_results[-1].model
            importance = dict(
                zip(FEATURE_NAMES, last_model.feature_importance(importance_type="gain"))
            )
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            print("\nTop 10 features (gain):")
            for name, val in sorted_imp[:10]:
                print(f"  {name}: {val:.1f}")

        return {"per_fold": [fr.metrics for fr in fold_results], "average": avg}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_metadata(
        self, params: dict, model: lgb.Booster, fold_results: list[FoldResult]
    ) -> dict:
        per_fold = [fr.metrics for fr in fold_results]
        avg = {}
        if per_fold:
            for k in per_fold[0]:
                avg[f"mean_{k}"] = float(np.mean([f[k] for f in per_fold]))

        importance = dict(
            zip(FEATURE_NAMES, model.feature_importance(importance_type="gain"))
        )

        return {
            "feature_names": FEATURE_NAMES,
            "horizon": self.horizon,
            "threshold_bps": self.threshold_bps,
            "training_start": self.start_datetime.isoformat(),
            "training_end": self.end_datetime.isoformat(),
            "params": params,
            "walk_forward_metrics": {**avg, "per_fold": per_fold},
            "feature_importance": importance,
        }
