from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    window: int = 60
    atr_multiple: float = 3.0


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local_1791244153"

    @classmethod
    def params_type(cls):
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        if params.window < 10 or not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0:
            raise ValueError("window must be >= 10 and atr_multiple must be positive and finite")
        return int(max(params.window + 6, 15))

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lag1"] = out["return"].shift(1)
        out["lag5"] = out["return"].shift(5)
        out["echo"] = out["return"].shift(4)
        out["rho1"] = out["return"].rolling(params.window, min_periods=params.window).corr(out["lag1"]).replace([np.inf, -np.inf], np.nan)
        out["rho5"] = out["return"].rolling(params.window, min_periods=params.window).corr(out["lag5"]).replace([np.inf, -np.inf], np.nan)
        out["forecast"] = out["rho1"] * out["return"] + out["rho5"] * out["echo"]
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["entry"] = (
            (out["rho1"] < 0) & (out["rho5"] > 0)
            & (out["return"] < 0) & (out["echo"] > 0)
            & (out["forecast"] > 0) & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: EchoParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i, (close, atr) in enumerate(zip(closes, atrs)):
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if active:
                if not valid:
                    active = False
                else:
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.atr_multiple * atr)
                    if close <= stop or i - entry_bar >= 18:
                        active = False
            elif valid and entries[i]:
                active = True
                entry_bar = i
                # The entry fills next bar; initialize the high from that bar's close.
                high_water = -np.inf
                stop = -np.inf
            raw[i] = int(active)
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = EchoParams(atr_multiple=2.0)
    index = pd.date_range("2020-01-01", periods=23, freq="B")
    data = pd.DataFrame({"close": [100.0, 101.0, 105.0, 101.0, 100.0] + [100.0] * 18}, index=index)
    features = pd.DataFrame({"atr": [1.0, 1.0, 1.0, 4.0] + [1.0] * 19, "entry": [True] + [False] * 22}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    signals = strategy.generate_signals(data, features, ctx, params).data
    assert signals["signal"].iloc[:5].tolist() == [0, 1, 1, 1, 0]
    data["close"] = 100.0
    features["atr"] = 1.0
    signals = strategy.generate_signals(data, features, ctx, params).data
    assert signals["signal"].iloc[1:19].eq(1).all()
    assert signals["signal"].iloc[19:].eq(0).all()
    assert signals["size"].gt(0).all()
    assert strategy.warmup_bars(params) == 66


if __name__ == "__main__":
    _self_check()
