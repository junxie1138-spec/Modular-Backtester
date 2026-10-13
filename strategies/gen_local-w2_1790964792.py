from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShelfParams:
    peak_bars: int = 20
    shelf_bars: int = 3
    atr_bars: int = 14
    min_depth: float = 0.015
    compression_atr: float = 1.6
    breakeven_pct: float = 0.006
    trail_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[ShelfParams]):
    strategy_id = "gen_local-w2_1790964792"

    @classmethod
    def params_type(cls):
        return ShelfParams

    def warmup_bars(self, params: ShelfParams) -> int:
        return int(max(200, params.peak_bars + 1, params.shelf_bars + 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: ShelfParams) -> pd.DataFrame:
        if min(params.peak_bars, params.shelf_bars, params.atr_bars) < 1 or not 3 <= params.hold_bars <= 5:
            raise ValueError("Lookbacks must be positive; hold_bars must be 3 through 5.")
        values = [params.min_depth, params.compression_atr, params.breakeven_pct, params.trail_atr]
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Thresholds and stop distances must be positive and finite.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["peak"] = close.rolling(params.peak_bars, min_periods=params.peak_bars).max()
        out["depth"] = 1.0 - close / out["peak"].replace(0.0, np.nan)
        out["shelf_depth"] = out["depth"].shift(1)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["shelf_high"] = data["high"].rolling(params.shelf_bars, min_periods=params.shelf_bars).max().shift(1)
        out["shelf_low"] = data["low"].rolling(params.shelf_bars, min_periods=params.shelf_bars).min().shift(1)
        out["shelf_span_atr"] = (out["shelf_high"] - out["shelf_low"]) / out["atr"].shift(1).replace(0.0, np.nan)
        ready = np.isfinite(out[["ma200", "shelf_depth", "atr", "shelf_high", "shelf_low", "shelf_span_atr"]]).all(axis=1)
        out["eligible"] = ready & close.gt(0) & out["atr"].gt(0) & out["shelf_depth"].ge(params.min_depth) & out["shelf_span_atr"].le(params.compression_atr)
        buy = out["eligible"] & close.gt(out["ma200"]) & close.gt(out["shelf_high"]) & out["depth"].lt(out["shelf_depth"])
        sell = out["eligible"] & close.lt(out["ma200"]) & close.lt(out["shelf_low"]) & out["depth"].gt(out["shelf_depth"])
        out["entry"] = np.select([buy, sell], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShelfParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["entry"].fillna(0).to_numpy(dtype=int)
        desired = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        pending, armed = False, False
        entry = best = stop = 0.0
        for i in range(len(data)):
            if side == 0:
                if np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                    side = int(candidates[i])
                    pending = side != 0
                    age = 0
                    armed = False
                desired[i] = side
                continue
            if pending:
                # The previous close's decision fills at this bar's open.
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    side = 0
                    pending = False
                    continue
                entry = opens[i]
                best = side * entry
                stop = best - params.trail_atr * atr[i - 1]
                pending = False
            age += 1
            if not np.isfinite(close[i]) or close[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                side = 0
                continue
            price = side * close[i]
            # Close-observed stops produce an exit at the following open.
            if price <= stop or age >= params.hold_bars:
                side = 0
                continue
            best = max(best, price)
            armed = armed or best - side * entry >= entry * params.breakeven_pct
            if armed:
                # Signed prices make both long and short protection ratchet upward.
                stop = max(stop, side * entry, best - params.trail_atr * atr[i])
            desired[i] = side
        df = pd.DataFrame({"signal": desired, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = ShelfParams(trail_atr=0.5)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": [100.0, 101.0, 100.4, 100.0, 100.0, 99.0, 99.6, 100.0], "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"atr": 1.0, "entry": [1, 0, 0, 0, -1, 0, 0, 0]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    ind["entry"] = [1, 0, 0, 0, 0, 0, 0, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert (result["size"] > 0).all()
