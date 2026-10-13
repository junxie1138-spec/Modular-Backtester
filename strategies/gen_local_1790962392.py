from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 3
    depth_threshold: float = 1.25
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1790962392"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(max(33, 16 + params.energy_window + 1))

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if params.energy_window < 1 or params.hold_bars < 1 or params.depth_threshold <= 0:
            raise ValueError("Windows, holding period and depth threshold must be positive")
        out = pd.DataFrame(index=data.index)
        price = np.log(data["close"].where(data["close"] > 0))
        out["log_price"] = price
        out["return"] = price.diff()
        out["volatility"] = out["return"].rolling(32, min_periods=32).std(ddof=0)
        for scale in (2, 4, 8):
            mean = price.rolling(scale, min_periods=scale).mean()
            out[f"mean_{scale}"] = mean
            # Causal Haar detail: newest block minus adjacent older block.
            out[f"haar_{scale}"] = np.sqrt(scale / 2.0) * (mean - mean.shift(scale))
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["fast_decay"] = (
            1.0 - out["energy_2"] / out["energy_2"].shift(1).where(out["energy_2"].shift(1) > 0)
        ).clip(0.0, 1.0)
        out["middle_growth"] = (
            1.0 - out["energy_4"].shift(1) / out["energy_4"].where(out["energy_4"] > 0)
        ).clip(0.0, 1.0)
        out["depth"] = -out["haar_8"] / (
            np.sqrt(4.0) * np.sqrt(8.0) * out["volatility"].where(out["volatility"] > 0)
        )
        ready = out.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        out["eligible"] = (
            ready
            & (out["haar_2"] < 0)
            & (out["haar_4"] < 0)
            & (out["haar_8"] < 0)
            & (out["fast_decay"] > 0)
            & (out["middle_growth"] > 0)
            & (out["depth"] >= params.depth_threshold)
        )
        out["conviction"] = (
            (out["depth"] / params.depth_threshold).clip(0.0, 2.0) / 2.0
            * np.sqrt(out["fast_decay"] * out["middle_growth"])
        ).clip(0.0, 1.0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        conviction = indicators["conviction"].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        exit_bar = -1
        entry_size = 1.0
        for i in range(len(data)):
            if i < exit_bar:
                signals[i] = 1
                sizes[i] = entry_size
            elif i == exit_bar:
                # Always emit a flat bar at expiry, even if entry conditions persist.
                continue
            elif eligible[i]:
                exit_bar = i + params.hold_bars
                entry_size = float(0.25 + 0.75 * np.clip(conviction[i], 0.0, 1.0))
                signals[i] = 1
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = WaveletParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=96, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    actual = strategy.indicators(data, params)
    assert np.isfinite(actual.to_numpy(dtype=float)).all()
    assert strategy.generate_signals(data, actual, ctx, params).data["signal"].eq(0).all()
    forced = pd.DataFrame({"eligible": True, "conviction": np.linspace(0.0, 1.0, len(data))}, index=index)
    result = strategy.generate_signals(data, forced, ctx, params).data
    n = params.hold_bars
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:n + 1].eq(1).all()
    assert result["signal"].iloc[n + 1] == 0
    assert result["signal"].iloc[n + 2] == 1
    assert result["size"].iloc[1:n + 1].eq(0.25).all()
    assert result["size"].iloc[n + 2] > 0.25
    assert result["size"].gt(0).all()
