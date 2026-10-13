from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    z_entry: float = 1.25
    trail_atr: float = 2.0

    def __post_init__(self):
        if not (np.isfinite(self.z_entry) and self.z_entry > 0):
            raise ValueError("z_entry must be finite and positive")
        if not (np.isfinite(self.trail_atr) and self.trail_atr > 0):
            raise ValueError("trail_atr must be finite and positive")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791211083"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 46

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["ma"] = close.rolling(40, min_periods=40).mean()
        out["std"] = close.rolling(40, min_periods=40).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["std"].replace(0, np.nan)
        out["dz"] = out["z"].diff()
        out["volume_mean"] = data["volume"].rolling(20, min_periods=20).mean()
        out["relative_volume"] = data["volume"].where(data["volume"] > 0) / out["volume_mean"].replace(0, np.nan)
        out["up_deformation"] = out["dz"].clip(lower=0).rolling(5, min_periods=5).sum()
        out["down_deformation"] = (-out["dz"].clip(upper=0)).rolling(5, min_periods=5).sum()
        out["up_volume"] = out["relative_volume"].where(out["dz"] > 0, 0.0).where(out["dz"].notna()).rolling(5, min_periods=5).sum()
        out["down_volume"] = out["relative_volume"].where(out["dz"] < 0, 0.0).where(out["dz"].notna()).rolling(5, min_periods=5).sum()
        out["recovery_efficiency"] = out["up_deformation"] / out["up_volume"].replace(0, np.nan)
        out["decline_efficiency"] = out["down_deformation"] / out["down_volume"].replace(0, np.nan)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["entry"] = (
            (out["z"].shift(1) <= -params.z_entry)
            & (out["z"] > -params.z_entry)
            & (out["z"] < 0)
            & (close > out["previous_close"])
            & (out["recovery_efficiency"] > 1.5 * out["decline_efficiency"])
            & (out["decline_efficiency"] > 0)
            & (out["atr"] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = pending = armed = False
        entry_price = stop = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if not (np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    continue
                # The previous close's decision fills at today's open.
                entry_price = opens[i]
                stop = entry_price - 2.0 * atr[i - 1]
                age = 0
                armed = False
                active = True
            if active:
                age += 1
                if not (np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0):
                    active = False
                    continue
                # Close-observed exits fill next open, like every other decision.
                if closes[i] <= stop or age >= 10:
                    active = False
                    continue
                if closes[i] >= 1.01 * entry_price:
                    armed = True
                if armed:
                    stop = max(stop, entry_price, closes[i] - params.trail_atr * atr[i])
                signals[i] = 1
            elif entries[i]:
                pending = True
                signals[i] = 1
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({
        "open": 100.0, "high": 105.0, "low": 96.0,
        "close": [100, 100, 102, 104, 101, 100, 100, 97, 100],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": 1.0,
        "entry": [True, False, False, False, False, True, False, False, False],
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
