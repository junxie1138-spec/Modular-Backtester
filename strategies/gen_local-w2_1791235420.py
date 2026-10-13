from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    drawdown_window: int = 20
    min_drawdown: float = 0.015
    atr_window: int = 14
    breakeven_gain: float = 0.004
    trail_atr: float = 0.8
    max_hold_bars: int = 2
    entry_slippage_bps: float = 5.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791235420"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(200, params.drawdown_window, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.drawdown_window < 2 or params.atr_window < 1
                or not 0 < params.min_drawdown < 1
                or params.breakeven_gain <= 0 or params.trail_atr <= 0
                or params.max_hold_bars not in (1, 2)
                or params.entry_slippage_bps < 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous = close.shift(1)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["peak"] = close.rolling(
            params.drawdown_window, min_periods=params.drawdown_window
        ).max()
        out["drawdown"] = 1.0 - close / out["peak"].where(out["peak"] > 0)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        month = pd.Series(data.index.to_period("M"), index=data.index)
        out["month_open"] = month.ne(month.shift(1)) & month.shift(1).notna()
        out["entry"] = (
            out["month_open"]
            & (out["drawdown"] >= params.min_drawdown)
            & (close > out["ma200"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        desired = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        pending = False
        active = False
        armed = False
        entry_price = peak = 0.0
        stop = -np.inf
        age = 0
        for i in range(len(data)):
            if pending:
                # Prior close's decision enters at this open; match YAML slippage.
                entry_price = opens[i] * (1.0 + params.entry_slippage_bps / 10000.0)
                active = np.isfinite(entry_price) and entry_price > 0
                peak = entry_price
                stop = -np.inf
                age = 0
                armed = False
                pending = False
            if active:
                age += 1
                valid = (
                    np.isfinite(highs[i]) and np.isfinite(closes[i])
                    and np.isfinite(atr[i]) and atr[i] > 0
                    and np.isfinite(regime[i])
                )
                if valid:
                    peak = max(peak, highs[i])
                    armed = armed or peak >= entry_price * (1.0 + params.breakeven_gain)
                    if armed:
                        stop = max(stop, entry_price, peak - params.trail_atr * atr[i])
                # Close-confirmed stops avoid assuming the high/low intrabar order.
                exit_now = (
                    not valid or age >= params.max_hold_bars
                    or closes[i] <= stop or closes[i] <= regime[i]
                )
                if exit_now:
                    active = False
                    continue
                desired[i] = 1
            elif entries[i]:
                desired[i] = 1
                pending = True
        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def self_check():
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    data = pd.DataFrame({
        "open": 100.0, "high": 100.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        "atr": 1.0, "ma200": 90.0,
        "entry": [True, False, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    params = Params(entry_slippage_bps=0.0)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    data.loc[index[1], "high"] = 101.0
    data.loc[index[1], "close"] = 100.1
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    self_check()
