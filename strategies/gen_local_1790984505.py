from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 4
    transfer_threshold: float = 0.55
    atr_window: int = 14
    activation_pct: float = 0.006
    atr_multiple: float = 1.2


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1790984505"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(max(200, params.energy_window + 7, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if (params.energy_window < 1 or params.atr_window < 1
                or not 0.0 < params.transfer_threshold < 1.0
                or not np.isfinite(params.activation_pct) or params.activation_pct <= 0
                or not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0):
            raise ValueError("Invalid wavelet or stop parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        # Causal, undecimated Haar details: adjacent blocks of 1, 2 and 4 closes.
        for scale in (1, 2, 4):
            block = out["log_close"].rolling(scale, min_periods=scale).mean()
            out[f"h{scale}"] = (block - block.shift(scale)) * np.sqrt(scale / 2.0)
            out[f"e{scale}"] = out[f"h{scale}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["fast_energy"] = out["e1"] + out["e2"]
        total = out["fast_energy"] + out["e4"]
        out["fast_share"] = out["fast_energy"] / total.where(total > 0)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["entry"] = (
            (close > out["ma200"])
            & (out["h1"] > 0) & (out["h2"] > 0) & (out["h4"] > 0)
            & (out["fast_share"] > params.transfer_threshold)
            & (out["fast_share"].shift(1) <= params.transfer_threshold)
            & (out["fast_energy"] > out["fast_energy"].shift(1))
            & (out["atr"] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: WaveletParams,
    ) -> SignalFrame:
        commands = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        armed = False
        entry = stop = 0.0
        age = 0
        for i in range(len(data)):
            row = data.iloc[i]
            atr = indicators["atr"].iloc[i]
            # Yesterday's target starts exposure at today's open.
            if not active and i > 0 and commands[i - 1] == 1:
                entry = float(row["open"])
                previous_atr = float(indicators["atr"].iloc[i - 1])
                if np.isfinite(entry) and entry > 0 and np.isfinite(previous_atr):
                    active = True
                    armed = False
                    age = 0
                    stop = entry - params.atr_multiple * previous_atr
            if active:
                age += 1
                valid = np.isfinite([row["high"], row["low"], row["close"], atr]).all()
                exit_now = not valid
                if valid:
                    # Test the existing stop first; a new ratchet starts at this close.
                    exit_now = row["low"] <= stop
                    if row["high"] >= entry * (1.0 + params.activation_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry, float(row["close"]) - params.atr_multiple * atr)
                    exit_now = exit_now or row["close"] <= stop
                exit_now = exit_now or age >= 2
                if exit_now:
                    active = False
                else:
                    commands[i] = 1
            elif entries[i]:
                commands[i] = 1
        df = pd.DataFrame({"signal": commands, "size": 0.95}, index=data.index)
        # Stop observations also execute at the next open, never retroactively intrabar.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = WaveletParams(activation_pct=0.01, atr_multiple=1.0)
    index = pd.date_range("2020-01-01", periods=5)
    bars = pd.DataFrame({
        "open": [100, 100, 101.5, 101, 101],
        "high": [100, 102, 102, 101, 101],
        "low": [100, 99.5, 100.4, 101, 101],
        "close": [100, 101.5, 101, 101, 101],
        "volume": [1000] * 5,
    }, index=index)
    features = pd.DataFrame({
        "atr": [1.0] * 5, "entry": [True, False, False, False, False]
    }, index=index)
    context = StrategyContext("SPY", "1d", 200)
    result = strategy.generate_signals(bars, features, context, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    # A first-session breakeven stop must exit at the following open.
    stopped = bars.copy()
    stopped.loc[index[1], "close"] = 99.8
    assert strategy.generate_signals(stopped, features, context, params).data[
        "signal"
    ].tolist() == [0, 1, 0, 0, 0]
    for end in range(1, len(bars) + 1):
        prefix = strategy.generate_signals(
            bars.iloc[:end], features.iloc[:end], context, params
        ).data
        assert prefix.equals(result.iloc[:end])
