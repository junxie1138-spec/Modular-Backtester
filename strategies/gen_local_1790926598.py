from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveEnergyParams:
    energy_window: int = 8
    coarse_share: float = 0.60
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[WaveEnergyParams]):
    strategy_id = "gen_local_1790926598"

    @classmethod
    def params_type(cls) -> type[WaveEnergyParams]:
        return WaveEnergyParams

    @staticmethod
    def warmup_bars(params: WaveEnergyParams) -> int:
        return int(max(200, params.atr_window, 8 + params.energy_window))

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveEnergyParams) -> pd.DataFrame:
        if params.energy_window < 1 or params.atr_window < 1 or params.hold_bars < 1:
            raise ValueError("Window lengths and holding period must be positive")
        if not 0.0 < params.coarse_share < 1.0 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("Invalid energy threshold or ATR stop multiplier")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["log_close"] = np.log(close.where(close > 0.0))
        out["return"] = out["log_close"].diff()
        # Causal Haar coefficients: recent versus preceding disjoint blocks.
        for half in (1, 2, 4):
            scale = 2 * half
            recent = out["return"].rolling(half, min_periods=half).mean()
            out[f"haar_{scale}"] = (recent - recent.shift(half)) * np.sqrt(half / 2.0)
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        total = out["energy_2"] + out["energy_4"] + out["energy_8"]
        out["coarse_share"] = out["energy_8"] / total.where(total > 0.0)
        out["slow_return"] = out["return"].rolling(8, min_periods=8).sum()
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["ma_200"] = close.rolling(200, min_periods=200).mean()
        ready = np.isfinite(out[["coarse_share", "slow_return", "haar_4", "haar_8", "atr", "ma_200"]]).all(axis=1)
        active = ready & (out["atr"] > 0.0) & (out["coarse_share"] >= params.coarse_share)
        long_entry = active & (close > out["ma_200"]) & (out["slow_return"] > 0.0) & (out["haar_4"] > 0.0) & (out["haar_8"] > 0.0)
        short_entry = active & (close < out["ma_200"]) & (out["slow_return"] < 0.0) & (out["haar_4"] < 0.0) & (out["haar_8"] < 0.0)
        out["candidate"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveEnergyParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        atrs = indicators["atr"].to_numpy(dtype=float)
        position = 0
        entry_bar = -1
        stop_distance = 0.0
        stop_price = np.nan
        for i in range(len(data)):
            if position:
                # Entry was decided yesterday; today's open is now observable.
                if i == entry_bar:
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        position = 0
                        continue
                    stop_price = opens[i] - position * stop_distance
                stopped = np.isfinite(closes[i]) and position * (closes[i] - stop_price) <= 0.0
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired or not np.isfinite(closes[i]):
                    position = 0
                    continue
            elif candidates[i] and np.isfinite(atrs[i]) and atrs[i] > 0.0:
                position = int(candidates[i])
                entry_bar = i + 1
                # Freeze the decision-bar ATR; never ratchet the stop.
                stop_distance = params.stop_atr * atrs[i]
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = WaveEnergyParams()
    index = pd.date_range("2020-01-01", periods=6)
    for direction, closes in ((1, [100, 105, 99, 97, 97, 97]), (-1, [100, 95, 101, 103, 103, 103])):
        data = pd.DataFrame({"open": 100.0, "close": closes}, index=index)
        indicators = pd.DataFrame({"candidate": [direction, 0, 0, 0, 0, 0], "atr": 1.0}, index=index)
        result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0, 0]
        assert (result["size"] > 0.0).all()
    params.hold_bars = 2
    data = pd.DataFrame({"open": 100.0, "close": 100.0}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, 0, 0, 0]
