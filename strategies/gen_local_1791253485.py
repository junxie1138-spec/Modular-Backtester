from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    box_bars: int = 5
    reference_bars: int = 20
    volume_bars: int = 20
    atr_bars: int = 14
    compression_ratio: float = 0.65
    volume_multiple: float = 1.4
    trail_atr: float = 1.25
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791253485"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.box_bars + params.reference_bars + 2,
                       params.volume_bars + 2, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.box_bars, params.reference_bars, params.volume_bars,
               params.atr_bars) < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not (0 < params.compression_ratio < 1):
            raise ValueError("compression_ratio must be between zero and one")
        if not np.isfinite(params.volume_multiple) or params.volume_multiple < 1:
            raise ValueError("volume_multiple must be finite and at least one")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be one or two")

        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["box_high"] = data["high"].rolling(params.box_bars).max()
        out["box_low"] = data["low"].rolling(params.box_bars).min()
        out["box_width"] = out["box_high"] - out["box_low"]
        out["width_baseline"] = out["box_width"].rolling(params.reference_bars).mean().shift(1)
        out["volume_baseline"] = data["volume"].rolling(params.volume_bars).mean().shift(1)
        out["launch_ceiling"] = out["box_high"].shift(1)
        out["confirmation_ceiling"] = out["box_high"].shift(2)
        out["compressed"] = (
            out["width_baseline"].gt(0)
            & out["box_width"].le(params.compression_ratio * out["width_baseline"])
        ).fillna(False)
        out["volume_confirmed"] = (
            out["volume_baseline"].gt(0)
            & data["volume"].ge(params.volume_multiple * out["volume_baseline"])
        ).fillna(False)
        out["launch"] = (
            out["compressed"].shift(1, fill_value=False)
            & data["close"].gt(out["launch_ceiling"])
            & data["close"].gt(out["previous_close"])
            & out["volume_confirmed"]
        ).fillna(False)
        out["entry"] = (
            out["launch"].shift(1, fill_value=False)
            & data["close"].gt(out["confirmation_ceiling"])
            & data["close"].gt(out["previous_close"])
            & out["volume_confirmed"]
            & out["atr"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        peak = -np.inf
        stop = -np.inf

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if active:
                if not valid:
                    active = False
                    continue
                # The first held bar is the session after the entry decision.
                held += 1
                peak = max(peak, close)
                stop = max(stop, peak - params.trail_atr * atr)
                if close <= stop or held >= params.max_hold_bars:
                    active = False
                else:
                    raw[i] = 1
            elif valid and entries[i]:
                active = True
                held = 0
                peak = -np.inf
                stop = -np.inf
                raw[i] = 1

        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = Params()
    index = pd.date_range("2020-01-01", periods=36)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0,
                         "close": 100.0, "volume": 100.0}, index=index)
    data.loc[index[20]:, "high"] = 100.5
    data.loc[index[20]:, "low"] = 99.5
    data.loc[index[30:32], "close"] = [103.0, 104.0]
    data.loc[index[30:32], "high"] = [103.5, 104.5]
    data.loc[index[30:32], "low"] = [100.0, 102.0]
    data.loc[index[30:32], "volume"] = [200.0, 180.0]
    derived = strategy.indicators(data, params)
    assert not derived["entry"].iloc[30] and derived["entry"].iloc[31]
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].iloc[31] == 0
    assert result["signal"].iloc[32:35].tolist() == [1, 1, 0]
    assert result["signal"].isin([0, 1]).all()
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= 27


if __name__ == "__main__":
    _self_check()
