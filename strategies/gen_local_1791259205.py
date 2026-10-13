from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalVolatilityParams:
    history_bars: int = 378
    volatility_bars: int = 21
    asymmetry_ratio: float = 1.1
    volatility_cap: float = 1.25
    minimum_samples: int = 10


class GeneratedStrategy(BaseStrategy[SeasonalVolatilityParams]):
    strategy_id = "gen_local_1791259205"

    @classmethod
    def params_type(cls):
        return SeasonalVolatilityParams

    def warmup_bars(self, params: SeasonalVolatilityParams) -> int:
        return int(max(params.history_bars, params.volatility_bars) + 2)

    def indicators(self, data: pd.DataFrame, params: SeasonalVolatilityParams) -> pd.DataFrame:
        if params.history_bars < 2 or params.volatility_bars < 2 or params.minimum_samples < 2:
            raise ValueError("Lookbacks and minimum_samples must be at least two.")
        if not np.isfinite(params.asymmetry_ratio) or params.asymmetry_ratio <= 0:
            raise ValueError("asymmetry_ratio must be finite and positive.")
        if not np.isfinite(params.volatility_cap) or params.volatility_cap <= 0:
            raise ValueError("volatility_cap must be finite and positive.")

        out = pd.DataFrame(index=data.index)
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["returns"] = returns
        out["upside_squared_return"] = returns.clip(lower=0).pow(2)
        out["downside_squared_return"] = returns.clip(upper=0).pow(2)
        out["recent_volatility"] = returns.rolling(
            params.volatility_bars, min_periods=params.volatility_bars
        ).std(ddof=0).shift(1)
        out["seasonal_upside_volatility"] = np.nan
        out["seasonal_downside_volatility"] = np.nan
        out["seasonal_samples"] = np.nan

        for month in range(1, 13):
            mask = data.index.month == month
            observed = pd.Series(mask, index=data.index) & returns.notna()
            count = observed.astype(float).rolling(
                params.history_bars, min_periods=params.history_bars
            ).sum().shift(1)
            denominator = count.replace(0, np.nan)
            out.loc[mask, "seasonal_samples"] = count.loc[mask]
            for source, target in (
                ("upside_squared_return", "seasonal_upside_volatility"),
                ("downside_squared_return", "seasonal_downside_volatility"),
            ):
                variance = out[source].where(observed, 0.0).rolling(
                    params.history_bars, min_periods=params.history_bars
                ).sum().shift(1).div(denominator)
                out.loc[mask, target] = np.sqrt(variance.clip(lower=0)).loc[mask]

        out["seasonal_volatility"] = np.sqrt(
            out["seasonal_upside_volatility"].pow(2)
            + out["seasonal_downside_volatility"].pow(2)
        )
        out["daily_condition"] = (
            (out["seasonal_samples"] >= params.minimum_samples)
            & (out["seasonal_upside_volatility"] > params.asymmetry_ratio * out["seasonal_downside_volatility"])
            & (out["recent_volatility"] > 0)
            & (out["recent_volatility"] <= params.volatility_cap * out["seasonal_volatility"])
        ).fillna(False)
        months = data.index.to_period("M")
        month_key = pd.Series(months, index=data.index)
        out["month_start"] = month_key.ne(month_key.shift(1))
        # The monthly condition updates only at the first observed session.
        # Consecutive favorable months retain the position until reversal.
        out["entry_condition"] = out["daily_condition"].astype(float).where(
            out["month_start"]
        ).groupby(months).ffill().fillna(0.0).astype(bool)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalVolatilityParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # Identical condition governs entry, continued exposure, and flat exit.
        df["signal"] = indicators["entry_condition"].fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.bdate_range("2018-01-01", periods=800)
    close = 100.0 * np.cumprod(1.0 + np.where(np.arange(800) % 2 == 0, 0.01, -0.002))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=index)
    strategy = GeneratedStrategy()
    params = SeasonalVolatilityParams(history_bars=252)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    features = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, features, ctx, params).data
    expected = features["entry_condition"].astype(int).shift(1).fillna(0).astype(int)
    assert signals["signal"].equals(expected)
    assert signals["signal"].sum() > 0
    assert set(signals["signal"].unique()) <= {0, 1}
    assert signals["size"].gt(0).all()
    assert features["entry_condition"].groupby(index.to_period("M")).nunique().max() == 1
    assert not features["entry_condition"].iloc[:strategy.warmup_bars(params) - 1].any()
    changed = data.copy()
    changed.iloc[-40:, changed.columns.get_loc("close")] *= 1.5
    pd.testing.assert_frame_equal(features.iloc[:-40], strategy.indicators(changed, params).iloc[:-40])
