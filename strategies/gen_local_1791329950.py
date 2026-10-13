from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReclaimParams:
    lookback: int = 12
    stop_atr: float = 0.8


class GeneratedStrategy(BaseStrategy[ReclaimParams]):
    strategy_id = "gen_local_1791329950"

    @classmethod
    def params_type(cls) -> type[ReclaimParams]:
        return ReclaimParams

    def warmup_bars(self, params: ReclaimParams) -> int:
        return int(max(params.lookback + 1, 15))

    def indicators(self, data: pd.DataFrame, params: ReclaimParams) -> pd.DataFrame:
        if not isinstance(params.lookback, int) or params.lookback < 2:
            raise ValueError("lookback must be an integer >= 2")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["typical_price"] = (data["high"] + data["low"] + data["close"]) / 3.0
        volume = data["volume"].where(data["volume"] > 0)
        out["volume_sum"] = volume.rolling(params.lookback).sum().shift(1)
        out["volume_mean"] = volume.rolling(params.lookback).mean().shift(1)
        out["weighted_price_sum"] = (out["typical_price"] * volume).rolling(params.lookback).sum().shift(1)
        out["anchor"] = out["weighted_price_sum"] / out["volume_sum"].replace(0, np.nan)
        out["previous_close"] = data["close"].shift(1)
        out["previous_open"] = data["open"].shift(1)
        out["previous_volume"] = volume.shift(1)
        out["session_range"] = data["high"] - data["low"]
        out["true_range"] = pd.concat([
            out["session_range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14).mean()
        out["close_location"] = (data["close"] - data["low"]) / out["session_range"].where(out["session_range"] > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["previous_close"] < out["anchor"])
            & (out["previous_close"] < out["previous_open"])
            & (out["previous_volume"] < 0.8 * out["volume_mean"])
            & (data["close"] > out["anchor"])
            & (data["close"] > data["open"])
            & (volume > 1.25 * out["volume_mean"])
            & (out["close_location"] >= 0.75)
            & out["atr"].notna()
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ReclaimParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        held = 0
        stop = np.nan
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    # Next-open entry includes the configured five-basis-point slippage.
                    entry_price = opens[i] * 1.0005
                    stop = entry_price - params.stop_atr * atr[i - 1]
                    active = True
                    held = 0
                else:
                    continue
            if active:
                held += 1
                if not np.isfinite(closes[i]) or closes[i] < stop or held >= 2:
                    active = False
                else:
                    signal[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                signal[i] = 1
                pending = True
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0, 100.0, 103.0, 101.0, 101.0, 100.0],
        "high": [104.0] * 6,
        "low": [98.0] * 6,
        "close": [100.0, 100.0, 101.0, 101.0, 99.0, 100.0],
        "volume": [1000.0] * 6,
    }, index=index)
    derived = pd.DataFrame({
        "atr": [1.0] * 6,
        "entry": [True, False, False, True, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, derived, None, ReclaimParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 0]
    assert result["size"].gt(0).all()
