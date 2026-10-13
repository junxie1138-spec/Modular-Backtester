from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FirstFridayParams:
    volume_multiple: float = 1.2
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[FirstFridayParams]):
    strategy_id = "gen_local-w2_1791258215"

    @classmethod
    def params_type(cls) -> type[FirstFridayParams]:
        return FirstFridayParams

    def warmup_bars(self, params: FirstFridayParams) -> int:
        return 64

    def indicators(self, data: pd.DataFrame, params: FirstFridayParams) -> pd.DataFrame:
        if not (np.isfinite(params.volume_multiple) and params.volume_multiple > 0
                and np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("Parameters must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["range"] = data["high"] - data["low"]
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["volume_baseline"] = data["volume"].rolling(63, min_periods=63).median().shift(1)
        out["volume_ratio"] = data["volume"].div(out["volume_baseline"].where(out["volume_baseline"] > 0))
        out["body"] = data["close"] - data["open"]
        out["body_fraction"] = out["body"].div(out["range"].where(out["range"] > 0))
        out["move_atr"] = (data["close"] - out["previous_close"]).div(out["atr"].where(out["atr"] > 0))
        out["first_friday"] = (data.index.dayofweek == 4) & (data.index.day <= 7)
        qualified = (
            out["first_friday"]
            & (out["volume_ratio"] >= params.volume_multiple)
            & (out["body_fraction"].abs() >= 0.6)
            & (out["move_atr"].abs() >= 0.35)
            & (out["body_fraction"] * out["move_atr"] > 0)
            & np.isfinite(out["atr"]) & (out["atr"] > 0)
        ).fillna(False)
        out["candidate"] = np.where(qualified, np.sign(out["body"].fillna(0)), 0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: FirstFridayParams,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        side = 0
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    # The next bar's open becomes known before this bar's close decision.
                    stop = opens[i] - side * params.stop_atr * atr[i - 1]
                stopped = (
                    not np.isfinite(stop) or not np.isfinite(closes[i])
                    or side * (closes[i] - stop) < 0
                )
                if stopped or i - entry_bar + 1 >= 4:
                    side = 0
                    continue
            elif candidates[i] and np.isfinite(atr[i]) and atr[i] > 0:
                side = int(candidates[i])
                entry_bar = i + 1
            signal[i] = side
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.bdate_range("2020-01-01", periods=9)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    strategy = GeneratedStrategy()
    params = FirstFridayParams()
    warm = strategy.indicators(data, params)
    assert warm["candidate"].eq(0).all()
    ind = pd.DataFrame({"candidate": [1] + [0] * 8, "atr": 2.0}, index=index)
    held = strategy.generate_signals(data, ind, None, params).data
    assert held["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0]
    data.loc[index[0], "close"] = 120.0
    data.loc[index[2], "close"] = 96.0
    stopped = strategy.generate_signals(data, ind, None, params).data
    assert stopped["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0]
    ind["candidate"] = [-1] + [0] * 8
    data.loc[index[2], "close"] = 104.0
    short = strategy.generate_signals(data, ind, None, params).data
    assert short["signal"].tolist() == [0, -1, -1, 0, 0, 0, 0, 0, 0]
    assert held["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
