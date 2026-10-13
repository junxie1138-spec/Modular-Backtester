from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    center_period: int = 5
    forecast_period: int = 20
    range_period: int = 8
    baseline_period: int = 40
    atr_period: int = 14
    compression_limit: float = 0.55
    innovation_floor: float = 0.03
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791330530"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(
            params.baseline_period + 1,
            params.center_period + params.forecast_period + 2,
            params.atr_period + 1,
            params.range_period,
        ) + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        windows = (
            params.center_period, params.forecast_period, params.range_period,
            params.baseline_period, params.atr_period,
        )
        if any(w < 2 for w in windows) or params.hold_bars < 1:
            raise ValueError("Lookbacks must be at least two; holding period must be positive.")
        if not (0 < params.compression_limit < 1 and params.innovation_floor >= 0
                and np.isfinite(params.innovation_floor)
                and np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("Invalid compression, innovation, or stop parameters.")

        out = pd.DataFrame(index=data.index)
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["latent_level"] = out["midpoint"].rolling(params.center_period).mean()
        out["latent_slope"] = out["latent_level"].diff()
        out["slope_forecast"] = out["latent_slope"].rolling(params.forecast_period).mean().shift(1)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        out["slope_innovation"] = (
            (out["latent_slope"] - out["slope_forecast"])
            / out["atr"].where(out["atr"] > 0)
        )
        out["short_width"] = (
            data["high"].rolling(params.range_period).max()
            - data["low"].rolling(params.range_period).min()
        )
        out["baseline_width"] = (
            data["high"].rolling(params.baseline_period).max()
            - data["low"].rolling(params.baseline_period).min()
        ).shift(1)
        out["compression"] = out["short_width"] / out["baseline_width"].where(out["baseline_width"] > 0)
        out["close_residual"] = data["close"] - out["midpoint"]
        out["residual_recovery"] = out["close_residual"].diff()
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        out["setup"] = (
            (out["compression"] <= params.compression_limit)
            & (out["latent_slope"] > 0)
            & (out["slope_innovation"] >= params.innovation_floor)
            & (out["close_residual"] < 0)
            & (out["residual_recovery"] > 0)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        setup = indicators["setup"].fillna(False).astype(bool)
        confirmed = (setup & setup.shift(1, fill_value=False)).to_numpy()
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        entry_atr = 0.0
        stop = 0.0
        held = 0

        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    # The prior close ordered entry; this bar's open is its entry reference.
                    stop = opens[i] - params.stop_atr * entry_atr
                    active = True
                    held = 0
            if active:
                held += 1
                stopped = np.isfinite(closes[i]) and closes[i] < stop
                if stopped or held >= params.hold_bars:
                    active = False
                else:
                    raw[i] = 1
                continue
            if confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0:
                entry_atr = atr[i]
                pending = True
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": [100.0] * 8,
        "high": [121.0] * 8,
        "low": [90.0] * 8,
        "close": [100.0, 100.0, 100.0, 120.0, 94.0, 94.0, 94.0, 94.0],
        "volume": [1000.0] * 8,
    }, index=index)
    indicators = pd.DataFrame({
        "setup": [True, True, False, False, False, False, False, False],
        "atr": [2.0, 2.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
    }, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    signals = strategy.generate_signals(data, indicators, ctx, GeneratedParams()).data
    assert signals["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    assert (signals["size"] > 0).all()
    data["close"] = 100.0
    timed = strategy.generate_signals(data, indicators, ctx, GeneratedParams(hold_bars=3)).data
    assert timed["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
