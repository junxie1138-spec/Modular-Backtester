from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_window: int = 7
    forecast_window: int = 4
    coupling_window: int = 12
    history_window: int = 84
    entry_percentile: float = 0.65


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790930527"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            params.slope_window + params.forecast_window + params.history_window,
            params.coupling_window + params.history_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.slope_window, params.forecast_window,
               params.coupling_window, params.history_window) < 2:
            raise ValueError("All windows must be at least two bars")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must lie strictly between zero and one")

        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        n = params.slope_window
        weights = np.arange(n, dtype=float) - (n - 1) / 2.0
        weights /= np.dot(weights, weights)
        slope = np.full(len(data), np.nan, dtype=float)
        if len(data) >= n:
            slope[n - 1:] = np.convolve(
                out["log_close"].to_numpy(), weights[::-1], mode="valid"
            )
        out["latent_slope"] = slope
        out["predicted_slope"] = out["latent_slope"].rolling(
            params.forecast_window, min_periods=params.forecast_window
        ).mean().shift(1)
        out["innovation"] = out["latent_slope"] - out["predicted_slope"]
        out["innovation_magnitude"] = out["innovation"].abs()
        out["innovation_threshold"] = out["innovation_magnitude"].shift(1).rolling(
            params.history_window, min_periods=params.history_window
        ).quantile(params.entry_percentile)

        open_price = data["open"].where(data["open"] > 0)
        close_price = data["close"].where(data["close"] > 0)
        out["overnight_return"] = np.log(open_price / close_price.shift(1))
        out["intraday_return"] = np.log(close_price / open_price)
        out["coupling"] = out["overnight_return"].rolling(
            params.coupling_window, min_periods=params.coupling_window
        ).corr(out["intraday_return"])
        out["coupling_threshold"] = out["coupling"].shift(1).rolling(
            params.history_window, min_periods=params.history_window
        ).quantile(0.5)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        required = ["innovation", "innovation_threshold", "coupling", "coupling_threshold"]
        valid = indicators[required].notna().all(axis=1)
        active = valid & (
            indicators["innovation_magnitude"] > indicators["innovation_threshold"]
        )
        reinforcing = indicators["coupling"] > indicators["coupling_threshold"]
        direction = np.sign(indicators["innovation"].fillna(0.0)).astype(int)
        candidate = direction.where(reinforcing, -direction).where(active, 0).astype(int)

        position = 0
        positions = np.zeros(len(data), dtype=int)
        for bar, entry in enumerate(candidate.to_numpy()):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            positions[bar] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    bars = np.arange(360, dtype=float)
    close = 100.0 * np.exp(0.0002 * bars + 0.025 * np.sin(bars / 7.0))
    opening = close * np.exp(0.004 * np.sin(bars / 3.0))
    sample = pd.DataFrame({
        "open": opening,
        "high": np.maximum(opening, close) * 1.005,
        "low": np.minimum(opening, close) * 0.995,
        "close": close,
        "volume": np.full(len(bars), 1000000.0),
    }, index=pd.date_range("2020-01-01", periods=len(bars), freq="B"))
    strategy = GeneratedStrategy()
    params = Params()
    features = strategy.indicators(sample, params)
    result = strategy.generate_signals(sample, features, None, params).data
    prefix = sample.iloc[:250]
    prefix_features = strategy.indicators(prefix, params)
    prefix_result = strategy.generate_signals(prefix, prefix_features, None, params).data
    pd.testing.assert_frame_equal(result.iloc[:250], prefix_result)
    assert result["signal"].isin([-1, 0, 1]).all()
    assert (result["size"] > 0).all()
    assert (result["signal"].iloc[:strategy.warmup_bars(params)] == 0).all()
    assert result["signal"].ne(0).any()
