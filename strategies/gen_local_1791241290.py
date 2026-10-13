from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    variance_window: int = 20
    block_days: int = 5
    strength_threshold: float = 1.3
    atr_window: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.0
    initial_stop_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791241290"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.variance_window + params.block_days + 1,
                       params.atr_window + 1, 2 * params.block_days + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.variance_window, params.block_days) < 2 or params.atr_window < 1:
            raise ValueError("Invalid indicator windows")
        if min(params.breakeven_pct, params.trail_atr, params.initial_stop_atr) <= 0 or params.max_hold < 1:
            raise ValueError("Invalid exit parameters")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["daily_return"] = out["log_close"].diff()
        out["block_return"] = out["log_close"].diff(params.block_days)
        out["drift"] = out["log_close"].diff(2 * params.block_days)
        out["daily_variance"] = out["daily_return"].rolling(params.variance_window).var(ddof=0)
        out["block_variance"] = out["block_return"].rolling(params.variance_window).var(ddof=0)
        denominator = params.block_days * out["daily_variance"]
        out["variance_ratio"] = (out["block_variance"] / denominator.where(denominator > 0)).replace([np.inf, -np.inf], np.nan)
        out["previous_ratio"] = out["variance_ratio"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["ma200"] = data["close"].rolling(200).mean()
        out["entry"] = (
            (out["variance_ratio"] > params.strength_threshold)
            & (out["previous_ratio"] <= params.strength_threshold)
            & (out["drift"] > 0)
            & (data["close"] > out["ma200"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        active = pending = armed = False
        entry_price = peak = stop = 0.0
        age = 0
        ohlc = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i, (opening, high, low, close) in enumerate(ohlc):
            valid = np.isfinite([opening, high, low, close, atr[i], regime[i]]).all()
            if not valid or min(opening, high, low, close, atr[i]) <= 0:
                active = pending = armed = False
                continue
            if pending:
                # The previous close's decision fills at this session's open.
                entry_price = peak = opening
                stop = entry_price - params.initial_stop_atr * atr[i - 1]
                age = 0
                active, pending, armed = True, False, False
            if active:
                age += 1
                # Existing stops are checked before today's high can raise them.
                # Stop breaches request next-open exits, never fills at the stop.
                if low <= stop or close <= regime[i] or age >= params.max_hold:
                    active = armed = False
                    continue
                peak = max(peak, high)
                if peak >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price, peak - params.trail_atr * atr[i])
                if close <= stop:
                    active = armed = False
                    continue
                decisions[i] = 1
            elif entries[i]:
                pending = True
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100, 100, 102, 104, 103, 102],
        "high": [101, 102, 104, 105, 104, 103],
        "low": [99, 99.5, 101, 102.5, 101.5, 101],
        "close": [100, 101.5, 103.5, 104, 102, 102],
        "volume": 1000,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1, 1, 1, 4, 4, 4],
        "ma200": 90.0,
        "entry": [True, False, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    params = Params()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    # ATR expansion cannot lower the stop raised to 102 on bar 2.
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    assert (result["size"] > 0).all()
    assert not strategy.indicators(data, params)["entry"].any()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
