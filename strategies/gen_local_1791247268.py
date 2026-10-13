from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    percentile_window: int = 126
    entry_percentile: float = 0.85
    atr_window: int = 14
    atr_multiple: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791247268"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.percentile_window + 12, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.percentile_window < 2 or params.atr_window < 1:
            raise ValueError("Lookbacks must be positive; percentile_window must be at least two")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between zero and one")
        if not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0 or params.max_hold < 1:
            raise ValueError("ATR multiple and holding period must be positive")
        out = pd.DataFrame(index=data.index)
        out["price_change"] = data["close"].diff()
        out["volume_change"] = data["volume"].diff()
        valid = np.isfinite(data[["open", "high", "low", "close", "volume"]]).all(axis=1)
        valid &= (data["close"] > 0) & (data["volume"] > 0) & (data["high"] >= data["low"])
        out["valid"] = valid
        out["contracting"] = (out["volume_change"] < 0) & valid & valid.shift(1, fill_value=False)
        for name, direction in (("up", 1), ("down", -1)):
            active = out["contracting"] & (direction * out["price_change"] > 0)
            out[name + "_active"] = active
            # Cap persistence at ten sessions; longer runs are already mature.
            out[name + "_streak"] = active.astype(int).groupby((~active).cumsum()).cumsum().clip(upper=10)
            out[name + "_threshold"] = out[name + "_streak"].shift(1).rolling(
                params.percentile_window, min_periods=params.percentile_window
            ).quantile(params.entry_percentile)
            out[name + "_qualified"] = out[name + "_streak"] > out[name + "_threshold"]
            out[name + "_trigger"] = out[name + "_qualified"] & ~out[name + "_qualified"].shift(1, fill_value=False)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs()
        ], axis=1).max(axis=1).where(valid & valid.shift(1, fill_value=False))
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        ready = valid & np.isfinite(out["atr"]) & (out["atr"] > 0)
        ready &= out[["up_threshold", "down_threshold"]].notna().all(axis=1)
        ready &= np.arange(len(data)) >= self.warmup_bars(params) - 1
        out["ready"] = ready
        out["entry"] = np.select(
            [ready & out["up_trigger"], ready & out["down_trigger"]], [1, -1], default=0
        ).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        side = 0
        age = 0
        high_water = low_water = stop = 0.0
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
                side = 0
                signals[i] = 0
                continue
            if side:
                age += 1
                high_water = max(high_water, close)
                low_water = min(low_water, close)
                if side == 1:
                    stop = max(stop, high_water - params.atr_multiple * atr)
                    stopped = close <= stop
                else:
                    stop = min(stop, low_water + params.atr_multiple * atr)
                    stopped = close >= stop
                if stopped or age >= params.max_hold:
                    side = 0
            elif entries[i]:
                side = int(entries[i])
                age = 0
                high_water = low_water = close
                stop = close - side * params.atr_multiple * atr
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    data = pd.DataFrame({"close": [100.0, 105.0, 101.0, 100.0, 95.0, 99.0, 99.0]},
                        index=pd.date_range("2020-01-01", periods=7))
    indicators = pd.DataFrame({"atr": [1.0, 1.0, 10.0, 1.0, 1.0, 10.0, 1.0],
                               "entry": [1, 0, 0, -1, 0, 0, 0]}, index=data.index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, Params(atr_multiple=2.0))
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert (result.data["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
