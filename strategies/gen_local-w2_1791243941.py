from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    breakeven_pct: float = 0.015
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791243941"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 24

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["return"] = data["close"].div(previous_close.replace(0, np.nan)).sub(1)
        out["volume_baseline"] = data["volume"].shift(1).rolling(20, min_periods=20).mean()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["impulse"] = (
            (out["return"] > 0.005)
            & (data["close"] > data["open"])
            & (data["volume"] > 1.5 * out["volume_baseline"])
        ).fillna(False)
        out["anchor_impulse"] = out["impulse"].shift(3, fill_value=False)
        out["anchor_high"] = data["high"].shift(3)
        out["anchor_low"] = data["low"].shift(3)
        out["anchor_midpoint"] = (data["high"] + data["low"]).shift(3) / 2.0
        out["anchor_volume"] = data["volume"].shift(3)
        out["pause_close_floor"] = data["close"].shift(1).rolling(2, min_periods=2).min()
        out["pause_low_floor"] = data["low"].shift(1).rolling(2, min_periods=2).min()
        out["pause_volume_ceiling"] = data["volume"].shift(1).rolling(2, min_periods=2).max()
        out["pause_volume_mean"] = data["volume"].shift(1).rolling(2, min_periods=2).mean()
        out["entry"] = (
            out["anchor_impulse"]
            & (out["pause_close_floor"] >= out["anchor_midpoint"])
            & (out["pause_low_floor"] >= out["anchor_low"])
            & (out["pause_volume_ceiling"] < 0.7 * out["anchor_volume"])
            & (data["close"] > out["anchor_high"])
            & (data["close"] > data["open"])
            & (data["volume"] > 1.2 * out["pause_volume_mean"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        if not np.isfinite(params.breakeven_pct) or params.breakeven_pct <= 0:
            raise ValueError("breakeven_pct must be positive and finite")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite")
        opens = data["open"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry_price = stop = peak_close = 0.0
        entry_bar = cooldown_until = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]):
                    active = True
                    armed = False
                    entry_price = peak_close = opens[i]
                    entry_bar = i
                    stop = entry_price - params.trail_atr * atr[i - 1]
            valid = (
                np.isfinite(opens[i]) and opens[i] > 0
                and np.isfinite(lows[i]) and lows[i] > 0
                and np.isfinite(closes[i]) and closes[i] > 0
                and np.isfinite(atr[i]) and atr[i] > 0
            )
            if active:
                # Test the previously established stop before raising it at this close.
                if not valid or opens[i] <= stop or lows[i] <= stop or i - entry_bar + 1 >= 10:
                    active = False
                    cooldown_until = i + 5
                    continue
                peak_close = max(peak_close, closes[i])
                if closes[i] >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price, peak_close - params.trail_atr * atr[i])
                signals[i] = 1
            elif valid and i >= cooldown_until and entries[i]:
                pending = True
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    data = pd.DataFrame({
        "open": [100.0, 100.0, 102.0, 103.0, 104.0, 103.0],
        "high": [101.0, 103.0, 104.0, 105.0, 105.0, 104.0],
        "low": [99.0, 99.0, 101.0, 102.0, 101.0, 102.0],
        "close": [100.0, 102.0, 103.0, 104.0, 103.0, 103.0],
        "volume": [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0] * 6,
        "entry": [True, False, False, False, True, True],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, StrategyParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    assert (result["size"] > 0).all()
