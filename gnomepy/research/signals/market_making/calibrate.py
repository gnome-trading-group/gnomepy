"""
Vol-model-aware calibrator for MarketMakingSignal.

Replays the actual VolatilityModel over historical data, then solves for
gamma / k that produce a user-specified target spread under the model's
real prediction distribution.

The AS spread formula ``gamma * sigma^2 / k + (2/gamma) * ln(1 + gamma/k)``
is quadratically sensitive to sigma, so even small distribution shifts between
a rolling-std calibrator and an LGBM vol model compound into large spread
changes. This calibrator closes that gap.
"""

import datetime

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

from gnomepy.data.cached_client import CachedMarketDataClient
from gnomepy.data.types import SchemaType
from gnomepy.registry.api import RegistryClient

from gnomepy.research.models.volatility import VolatilityModel


class MarketMakingCalibrator:
    """Calibrate MarketMakingSignal parameters against a concrete VolatilityModel.

    Parameters
    ----------
    volatility_model : VolatilityModel
        The actual model used at runtime (e.g. LGBMVolatilityModel).
    listing_id : int
        Listing to calibrate for.
    start_datetime, end_datetime : datetime.datetime
        Historical window to replay.
    schema_type : SchemaType
        Market data schema.
    market_data_client : CachedMarketDataClient, optional
    registry_client : RegistryClient, optional
    """

    def __init__(
        self,
        volatility_model: VolatilityModel,
        listing_id: int,
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
        schema_type: SchemaType = SchemaType.MBP_10,
        market_data_client: CachedMarketDataClient | None = None,
        registry_client: RegistryClient | None = None,
    ):
        self.volatility_model = volatility_model
        self.listing_id = listing_id
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.schema_type = schema_type

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

        # Lazily populated
        self._raw_df: pd.DataFrame | None = None
        self._col_arrays: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Fetch MBP10 data and compute midPrice."""
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

        # Pre-compute column arrays for fast slice-view replay
        self._col_arrays = {col: df[col].values for col in df.columns}

        print(f"  Loaded {len(df):,} rows")
        return df

    # ------------------------------------------------------------------
    # Vol model replay
    # ------------------------------------------------------------------

    def run_vol_model(
        self,
        stride: int = 10,
        max_lookback: int | None = None,
    ) -> np.ndarray:
        """Replay the volatility model over historical data.

        Builds ``listing_data`` dicts using numpy slice views (O(1) per
        column, no copy) to simulate OMS behaviour.

        Parameters
        ----------
        stride : int
            Sample every *stride* ticks (default 10).
        max_lookback : int, optional
            Max history fed to the model per tick. Defaults to
            ``volatility_model.min_lookback``.

        Returns
        -------
        np.ndarray
            Array of predicted bps values (NaN where model returned None).
        """
        if self._col_arrays is None:
            self.load_data()
        col_arrays = self._col_arrays
        n = len(next(iter(col_arrays.values())))

        if max_lookback is None:
            max_lookback = self.volatility_model.min_lookback

        min_start = self.volatility_model.min_lookback
        indices = range(min_start, n, stride)
        predictions = np.full(len(indices), np.nan)

        for j, i in enumerate(indices):
            start = max(0, i + 1 - max_lookback)
            listing_data = {col: arr[start:i + 1] for col, arr in col_arrays.items()}
            pred = self.volatility_model.predict(listing_data)
            if pred is not None:
                predictions[j] = pred

        valid = ~np.isnan(predictions)
        print(
            f"  Vol model replay: {int(valid.sum()):,} / {len(predictions):,} "
            f"valid predictions (stride={stride})"
        )
        return predictions[valid]

    # ------------------------------------------------------------------
    # Order arrival rate
    # ------------------------------------------------------------------

    def estimate_order_arrival_rate(self, window: int = 1000) -> float:
        """Spread-based estimate of order arrival rate k.

        ``k = 1 / (relative_spread * 10 + eps)``
        """
        if self._raw_df is None:
            self.load_data()
        df = self._raw_df

        spreads = df["askPrice0"] - df["bidPrice0"]
        relative_spreads = spreads / df["midPrice"]
        mean_spread = float(np.nanmean(relative_spreads))

        k = 1.0 / (mean_spread * 10.0 + 1e-6)
        return k

    # ------------------------------------------------------------------
    # AS spread formula (mirrors signal.py:260-274)
    # ------------------------------------------------------------------

    @staticmethod
    def _as_spread(gamma: float, sigma: float, k: float) -> float:
        """Avellaneda-Stoikov optimal spread at zero inventory.

        ``s = gamma * sigma^2 / k + (2/gamma) * ln(1 + gamma/k)``
        """
        if gamma <= 0 or k <= 0 or sigma <= 0:
            return 0.0
        c1 = gamma * (sigma ** 2) / k
        log_arg = 1.0 + gamma / k
        c2 = (2.0 / gamma) * np.log(log_arg) if log_arg > 0 else 0.0
        return max(0.0, c1 + c2)

    # ------------------------------------------------------------------
    # Gamma solver
    # ------------------------------------------------------------------

    def calibrate_gamma(
        self,
        target_spread_bps: float,
        k: float,
        sigma_at_quantile: float,
        mid_price: float,
    ) -> float:
        """Find gamma producing *target_spread_bps* at the given sigma.

        The AS spread as a function of gamma is U-shaped. We:
        1. Find ``gamma_min`` (the minimum of the curve) via bounded
           minimisation.
        2. If the target is below the achievable minimum, warn and
           return ``gamma_min``.
        3. Otherwise, solve on the **right branch** (larger gamma =
           proper inventory risk aversion) using Brent's method.

        Parameters
        ----------
        target_spread_bps : float
            Desired spread in basis points.
        k : float
            Order arrival rate.
        sigma_at_quantile : float
            Per-tick sigma (log-return units) at the chosen quantile.
        mid_price : float
            Representative mid price for bps conversion.

        Returns
        -------
        float
            Calibrated gamma.
        """
        target_spread_abs = mid_price * (target_spread_bps / 1e4)

        if sigma_at_quantile <= 0:
            raise ValueError(
                f"sigma_at_quantile is {sigma_at_quantile} — the volatility "
                f"model is predicting zero volatility at the chosen quantile. "
                f"Check that the model produces non-zero predictions."
            )

        def spread_fn(gamma: float) -> float:
            return self._as_spread(gamma, sigma_at_quantile, k)

        # For large gamma the spread ≈ gamma * sigma^2 / k, so we need
        # gamma ≈ target * k / sigma^2 to reach the target.  Use 2× as
        # a safe upper bound for both the minimiser and brentq.
        gamma_upper = max(1e10, 2.0 * target_spread_abs * k / (sigma_at_quantile ** 2))

        # 1. Find minimum of the spread curve
        result = minimize_scalar(
            spread_fn, bounds=(1e-6, gamma_upper), method="bounded",
            options={"xatol": 1e-8, "maxiter": 500},
        )
        gamma_min = result.x
        spread_min = spread_fn(gamma_min)

        if target_spread_abs <= spread_min:
            print(
                f"  WARNING: target spread {target_spread_bps:.2f} bps "
                f"({target_spread_abs:.6f}) is below minimum achievable "
                f"spread ({spread_min / mid_price * 1e4:.2f} bps). "
                f"Using gamma at minimum spread."
            )
            return gamma_min

        # 2. Solve on the right branch [gamma_min, gamma_upper]
        def root_fn(gamma: float) -> float:
            return spread_fn(gamma) - target_spread_abs

        gamma = brentq(root_fn, gamma_min, gamma_upper, xtol=1e-6, maxiter=500)
        return gamma

    # ------------------------------------------------------------------
    # Full calibration pipeline
    # ------------------------------------------------------------------

    def calibrate(
        self,
        target_spread_bps: float = 1.0,
        spread_quantile: float = 0.5,
        cb_activation_rate: float = 0.01,
        stride: int = 10,
        min_volatility: float = 1e-8,
    ) -> dict:
        """Run the full calibration pipeline.

        Parameters
        ----------
        target_spread_bps : float
            Desired spread in bps at the chosen quantile of vol predictions.
        spread_quantile : float
            Quantile of the prediction distribution to calibrate to
            (0.5 = median).
        cb_activation_rate : float
            Fraction of ticks where the circuit breaker should activate.
            E.g. 0.01 → P99 threshold.
        stride : int
            Sample every *stride* ticks when replaying the vol model.
        min_volatility : float
            Per-tick sigma floor. Must match the ``min_volatility`` used by
            ``MarketMakingSignal`` at runtime, otherwise the calibrated gamma
            will be inconsistent with the actual spread produced by the
            signal.

        Returns
        -------
        dict
            Calibration results (see class docstring for keys).
        """
        # 1. Load data
        df = self.load_data()
        n = len(df)

        # 2. Run vol model
        predictions_bps = self.run_vol_model(stride=stride)
        if len(predictions_bps) == 0:
            raise RuntimeError("Vol model produced no valid predictions.")

        # 3. Convert per-tick bps predictions to sigma (log-return units),
        #    applying the same min_volatility floor that MarketMakingSignal uses.
        horizon = self.volatility_model.horizon
        sigma_per_tick = predictions_bps / 1e4
        sigma_per_tick = np.maximum(sigma_per_tick, min_volatility)

        # 4. Estimate k from spreads
        k = self.estimate_order_arrival_rate()
        print(f"  Order arrival rate (k): {k:,.0f}")

        # 5. Solve for gamma
        sigma_q = float(np.quantile(sigma_per_tick, spread_quantile))
        mid_price = float(np.nanmedian(df["midPrice"].values))

        gamma = self.calibrate_gamma(target_spread_bps, k, sigma_q, mid_price)
        print(f"  Calibrated gamma: {gamma:,.0f}")

        # 6. Circuit breaker threshold
        cb_percentile = (1.0 - cb_activation_rate) * 100.0
        vol_threshold_bps = float(np.percentile(predictions_bps, cb_percentile))

        # 7. Variance ratio diagnostic
        mid_prices = df["midPrice"].values
        log_returns = np.diff(np.log(mid_prices))
        h = horizon
        if len(log_returns) >= h:
            var_1 = float(np.var(log_returns, ddof=1))
            # h-tick returns
            h_returns = np.log(mid_prices[h:]) - np.log(mid_prices[:-h])
            var_h = float(np.var(h_returns, ddof=1))
            variance_ratio = var_h / (h * var_1) if var_1 > 0 else np.nan
        else:
            variance_ratio = np.nan

        if not np.isnan(variance_ratio):
            if abs(variance_ratio - 1.0) > 0.3:
                print(
                    f"  WARNING: Variance ratio = {variance_ratio:.2f} "
                    f"(far from 1.0 — returns are not a random walk at "
                    f"horizon {h})"
                )

        # 8. Prediction stats
        pct = [5, 10, 25, 50, 75, 90, 95, 99]
        prediction_stats = {
            "mean_bps": float(np.mean(predictions_bps)),
            "median_bps": float(np.median(predictions_bps)),
        }
        for p in pct:
            prediction_stats[f"p{p}_bps"] = float(np.percentile(predictions_bps, p))

        # 9. Spread at various quantiles (zero inventory)
        spread_at_quantiles = {}
        for q_label, q_val in [("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p95", 0.95)]:
            sigma_at_q = float(np.quantile(sigma_per_tick, q_val))
            spread_abs = self._as_spread(gamma, sigma_at_q, k)
            spread_bps = (spread_abs / mid_price) * 1e4
            spread_at_quantiles[q_label] = round(spread_bps, 4)

        results = {
            "gamma": gamma,
            "order_arrival_rate": k,
            "vol_threshold_bps": vol_threshold_bps,
            "min_volatility": min_volatility,
            "prediction_stats": prediction_stats,
            "spread_at_quantiles": spread_at_quantiles,
            "variance_ratio": variance_ratio,
            "n_predictions": len(predictions_bps),
            "n_ticks": n,
            "mid_price": mid_price,
            "horizon": horizon,
            "target_spread_bps": target_spread_bps,
            "spread_quantile": spread_quantile,
            "cb_activation_rate": cb_activation_rate,
        }
        return results

    # ------------------------------------------------------------------
    # Spread distribution
    # ------------------------------------------------------------------

    def spread_distribution(
        self,
        gamma: float,
        k: float,
        inventory: float = 0.0,
        stride: int = 10,
        min_volatility: float = 1e-8,
    ) -> pd.DataFrame:
        """Compute the spread at every sampled tick for analysis.

        Parameters
        ----------
        gamma : float
            Risk aversion.
        k : float
            Order arrival rate.
        inventory : float
            Fixed inventory level (default 0).
        stride : int
            Sample stride for vol model replay.
        min_volatility : float
            Per-tick sigma floor. Must match ``MarketMakingSignal.min_volatility``.

        Returns
        -------
        pd.DataFrame
            Columns: predicted_bps, sigma, spread_abs, spread_bps
        """
        predictions_bps = self.run_vol_model(stride=stride)
        sigma = predictions_bps / 1e4
        sigma = np.maximum(sigma, min_volatility)

        mid_price = float(np.nanmedian(self._raw_df["midPrice"].values))

        spread_abs = np.array([
            self._as_spread(gamma, s, k) for s in sigma
        ])
        spread_bps = (spread_abs / mid_price) * 1e4

        return pd.DataFrame({
            "predicted_bps": predictions_bps,
            "sigma": sigma,
            "spread_abs": spread_abs,
            "spread_bps": spread_bps,
        })

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self, results: dict) -> None:
        """Pretty-print a calibration summary."""
        model_name = type(self.volatility_model).__name__
        horizon = results["horizon"]
        n_ticks = results["n_ticks"]
        n_preds = results["n_predictions"]
        stats = results["prediction_stats"]
        spreads = results["spread_at_quantiles"]
        vr = results["variance_ratio"]
        target = results["target_spread_bps"]

        print(f"\n{'=' * 40}")
        print(f"  MarketMaking Calibration Report")
        print(f"{'=' * 40}")
        print(f"Vol Model: {model_name} (horizon={horizon})")
        print(f"Data: {n_ticks:,} ticks ({n_preds:,} sampled)")
        print()
        print("Prediction Distribution (per-tick bps):")
        print(
            f"  Median: {stats['median_bps']:.2f}   "
            f"Mean: {stats['mean_bps']:.2f}   "
            f"P95: {stats['p95_bps']:.2f}   "
            f"P99: {stats['p99_bps']:.2f}"
        )
        print()
        print("Calibrated Parameters:")
        print(f"  gamma:              {results['gamma']:,.0f}")
        print(f"  order_arrival_rate: {results['order_arrival_rate']:,.0f}")
        cb_pct = (1.0 - results["cb_activation_rate"]) * 100
        print(
            f"  vol_threshold_bps:  {results['vol_threshold_bps']:.2f} "
            f"(P{cb_pct:.0f}, ~{results['cb_activation_rate']*100:.0f}% CB rate)"
        )
        print()
        print("Spread Profile (zero inventory):")
        print(
            f"  At P25 vol: {spreads['p25']:.2f} bps    "
            f"At median vol: {spreads['p50']:.2f} bps [TARGET={target:.2f}]"
        )
        print(
            f"  At P75 vol: {spreads['p75']:.2f} bps    "
            f"At P95 vol:    {spreads['p95']:.2f} bps"
        )
        print()
        if not np.isnan(vr):
            status = "OK" if abs(vr - 1.0) <= 0.3 else "WARNING"
            print(f"Variance Ratio: {vr:.2f} ({status})")
        else:
            print("Variance Ratio: N/A")
        print()

    # ------------------------------------------------------------------
    # Recommended parameters
    # ------------------------------------------------------------------

    def get_recommended_parameters(
        self,
        max_inventory: float = 100.0,
        target_spread_bps: float = 1.0,
        spread_quantile: float = 0.5,
        cb_activation_rate: float = 0.01,
        stride: int = 10,
        liquidation_threshold: float = 0.8,
        min_volatility: float = 1e-8,
    ) -> dict:
        """Run calibration and return a dict ready for ``MarketMakingSignal(**params)``.

        Parameters
        ----------
        max_inventory : float
        target_spread_bps : float
        spread_quantile : float
        cb_activation_rate : float
        stride : int
        liquidation_threshold : float
        min_volatility : float
            Per-tick sigma floor. Must match ``MarketMakingSignal.min_volatility``.

        Returns
        -------
        dict
            Keys match ``MarketMakingSignal.__init__`` kwargs.
        """
        results = self.calibrate(
            target_spread_bps=target_spread_bps,
            spread_quantile=spread_quantile,
            cb_activation_rate=cb_activation_rate,
            stride=stride,
            min_volatility=min_volatility,
        )
        self.report(results)

        return {
            "gamma": results["gamma"],
            "order_arrival_rate": results["order_arrival_rate"],
            "vol_threshold_bps": results["vol_threshold_bps"],
            "min_volatility": min_volatility,
            "max_inventory": max_inventory,
            "liquidation_threshold": liquidation_threshold,
        }
