from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    trend_window: int = 12
    volatility_window: int = 20
    innovation_threshold: float = 0.6


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791266793"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(200, params.trend_window + 1, params.volatility_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.trend_window < 2 or params.volatility_window < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not np.isfinite(params.innovation_threshold) or params.innovation_threshold < 0:
            raise ValueError("Innovation threshold must be finite and nonnegative")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out["log_close"] = np.log(close.where(close > 0))
        out["log_return"] = out["log_close"].diff()
        out["regime_ma"] = close.rolling(200, min_periods=200).mean()
        out["return_volatility"] = out["log_return"].rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=0)
        w = params.trend_window
        out["time_coordinate"] = np.arange(len(data), dtype=float)
        out["log_sum"] = out["log_close"].rolling(w, min_periods=w).sum()
        out["time_log_sum"] = (out["time_coordinate"] * out["log_close"]).rolling(w, min_periods=w).sum()
        out["time_mean"] = out["time_coordinate"] - (w - 1) / 2.0
        out["latent_slope"] = (out["time_log_sum"] - out["time_mean"] * out["log_sum"]) / (w * (w * w - 1) / 12.0)
        out["latent_endpoint"] = out["log_sum"] / w + out["latent_slope"] * (w - 1) / 2.0
        out["forecast"] = out["latent_endpoint"].shift(1) + out["latent_slope"].shift(1)
        out["forecast_innovation"] = out["log_close"] - out["forecast"]
        # Endpoint leverage maps the forecast surprise into a slope update.
        gain = 6.0 / (w * (w + 1))
        out["slope_innovation"] = gain * out["forecast_innovation"]
        out["normalized_innovation"] = out["slope_innovation"] / (gain * out["return_volatility"].where(out["return_volatility"] > 0))
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        eligible = (
            (data["close"] > indicators["regime_ma"])
            & (indicators["latent_slope"] > 0)
            & (indicators["normalized_innovation"] > params.innovation_threshold)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Maintain exposure only while the entry condition holds; its reversal exits.
        df["signal"] = eligible.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = Params()
    index = pd.date_range("2020-01-01", periods=240, freq="B")
    price = 100.0 * np.exp(np.arange(240) * 0.001)
    data = pd.DataFrame({"open": price, "high": price * 1.01, "low": price * 0.99, "close": price, "volume": 1000000.0}, index=index)
    derived = GeneratedStrategy.indicators(data, p)
    assert np.nanmax(np.abs(derived["latent_slope"] - 0.001)) < 1e-10
    assert np.nanmax(np.abs(derived["forecast_innovation"])) < 1e-10
    probe = pd.DataFrame({"regime_ma": 0.0, "latent_slope": 0.001, "normalized_innovation": [1.0, 1.0, -1.0, 1.0, -1.0]}, index=index[:5])
    result = GeneratedStrategy.generate_signals(data.iloc[:5], probe, None, p).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1]
    assert (result["size"] > 0).all()
    assert GeneratedStrategy.warmup_bars(p) >= 200
