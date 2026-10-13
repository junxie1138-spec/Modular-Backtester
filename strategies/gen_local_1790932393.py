from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    slope_window: int = 10
    echo_horizon: int = 18
    correlation_window: int = 84
    min_correlation: float = 0.20
    entry_threshold: float = 0.08


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790932393"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(200, params.slope_window + params.echo_horizon + params.correlation_window + 2)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.slope_window, params.echo_horizon, params.correlation_window) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not 0 < params.min_correlation < 1 or params.entry_threshold <= 0:
            raise ValueError("Correlation must lie in (0, 1); threshold must be positive.")
        out = pd.DataFrame(index=data.index)
        out["log_return"] = np.log(data["close"].where(data["close"] > 0)).diff()
        out["latent_slope"] = out["log_return"].rolling(params.slope_window).mean()
        out["innovation"] = out["latent_slope"].diff()
        out["lagged_innovation"] = out["innovation"].shift(params.echo_horizon)
        out["echo_correlation"] = out["innovation"].rolling(params.correlation_window).corr(out["lagged_innovation"])
        out["innovation_sum"] = out["innovation"].rolling(params.echo_horizon).sum()
        # Each of the next H slope innovations is forecast from its H-bar echo.
        out["projected_slope"] = out["latent_slope"] + out["echo_correlation"] * out["innovation_sum"]
        out["volatility"] = out["log_return"].rolling(params.echo_horizon).std(ddof=0)
        out["forecast_score"] = out["projected_slope"] / out["volatility"].where(out["volatility"] > 0)
        out["regime_ma"] = data["close"].rolling(200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        score = indicators["forecast_score"]
        ma = indicators["regime_ma"]
        ready = score.notna() & ma.notna() & indicators["echo_correlation"].ge(params.min_correlation)
        long_entry = ready & score.gt(params.entry_threshold) & data["close"].gt(ma)
        short_entry = ready & score.lt(-params.entry_threshold) & data["close"].lt(ma)
        candidate = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(candidate):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                # An opposite eligible entry closes the position; stay flat this bar.
                position = 0
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [101., 101., 101., 99., 99., 101., 101.]}, index=index)
    derived = pd.DataFrame({
        "forecast_score": [0.2, 0.2, 0., -0.2, -0.2, 0.2, 0.2],
        "regime_ma": 100.,
        "echo_correlation": 0.5,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, derived, ctx, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, -1, 0]
    flat = pd.DataFrame(100., index=pd.date_range("2020-01-01", periods=240), columns=["open", "high", "low", "close", "volume"])
    result = strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params)
    assert result.data["signal"].eq(0).all() and result.data["size"].gt(0).all()
