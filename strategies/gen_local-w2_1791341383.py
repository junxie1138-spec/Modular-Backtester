from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 5
    atr_period: int = 14
    dominance: float = 0.72
    breakeven_gain: float = 0.012
    atr_multiple: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791341383"

    @classmethod
    def params_type(cls):
        return WaveletParams

    @staticmethod
    def warmup_bars(params):
        return int(max(16 + params.energy_window - 1, params.atr_period + 1))

    @staticmethod
    def indicators(data, params):
        if (params.energy_window < 1 or params.atr_period < 1
                or not 0 < params.dominance < 1
                or params.breakeven_gain <= 0
                or params.atr_multiple <= 0 or params.max_hold < 1):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        log_close = np.log(data["close"].where(data["close"] > 0))
        out["log_close"] = log_close
        for scale in (2, 4, 8):
            mean = log_close.rolling(scale, min_periods=scale).mean()
            out[f"mean_{scale}"] = mean
            coefficient = np.sqrt(scale / 2.0) * (mean - mean.shift(scale))
            out[f"haar_{scale}"] = coefficient
            out[f"energy_{scale}"] = (coefficient.pow(2) / scale ** 2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["total_energy"] = out[["energy_2", "energy_4", "energy_8"]].sum(
            axis=1, min_count=3
        )
        out["coarse_fraction"] = (
            out["energy_4"] + out["energy_8"]
        ) / out["total_energy"].where(out["total_energy"] > 0)
        previous_close = data["close"].shift(1)
        out["previous_close"] = previous_close
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        out["setup"] = (
            out["coarse_fraction"].ge(params.dominance)
            & out["haar_2"].gt(0)
            & out["haar_4"].gt(0)
            & out["haar_8"].gt(0)
            & out["atr"].gt(0)
        ).fillna(False)
        out["confirmed"] = out["setup"] & out["setup"].shift(1, fill_value=False)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx: StrategyContext, params):
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        pending = False
        holding = False
        activated = False
        entry = peak = stop = np.nan
        age = 0
        for i in range(len(data)):
            exited = False
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry = opens[i]
                    peak = entry
                    stop = entry - params.atr_multiple * atrs[i - 1]
                    activated = False
                    holding = True
                    age = 0
            if holding:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0:
                    decisions[i] = 1
                    continue
                peak = max(peak, closes[i])
                activated = activated or closes[i] >= entry * (1 + params.breakeven_gain)
                if activated:
                    stop = max(stop, entry)
                    if np.isfinite(atrs[i]) and atrs[i] > 0:
                        stop = max(stop, peak - params.atr_multiple * atrs[i])
                if closes[i] <= stop or age >= params.max_hold:
                    holding = False
                    exited = True
                else:
                    decisions[i] = 1
            if not holding and not exited and confirmed[i]:
                if np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atrs[i]):
                    decisions[i] = 1
                    pending = True
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=100, freq="B")
    close = 100 * np.exp(np.arange(100) * 0.003)
    data = pd.DataFrame({
        "open": close / np.exp(0.003), "high": close * 1.005,
        "low": close * 0.995, "close": close, "volume": 1000000.0,
    }, index=index)
    params = WaveletParams()
    indicators = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    signal = result["signal"]
    assert signal.isin([0, 1]).all() and result["size"].gt(0).all()
    first_confirmation = np.flatnonzero(indicators["confirmed"].to_numpy())[0]
    assert np.flatnonzero(signal.to_numpy())[0] == first_confirmation + 1
    groups = signal.ne(signal.shift()).cumsum()
    assert signal.groupby(groups).sum().max() <= params.max_hold
    assert GeneratedStrategy.warmup_bars(params) >= 20


if __name__ == "__main__":
    _self_check()
