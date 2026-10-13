from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_bars: int = 3
    breadth_bars: int = 12
    history_bars: int = 126
    tail_quantile: float = 0.8
    saturation_quantile: float = 0.7
    atr_bars: int = 14
    breakeven_pct: float = 0.01
    trail_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791220094"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.history_bars + max(params.breadth_bars + 1, params.roc_bars + 2, params.atr_bars + 1) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        daily = close.pct_change(fill_method=None)
        out["up_share"] = daily.gt(0).astype(float).where(daily.notna()).rolling(params.breadth_bars).mean()
        history = out["acceleration"].shift(1).rolling(params.history_bars)
        out["accel_upper"] = history.quantile(params.tail_quantile)
        out["accel_lower"] = history.quantile(1.0 - params.tail_quantile)
        out["saturation"] = out["up_share"].shift(1).rolling(params.history_bars).quantile(params.saturation_quantile)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1) & out["atr"].gt(0)
        saturated = out["up_share"].gt(out["saturation"])
        propagation = (~saturated) & out["roc"].gt(0) & out["acceleration"].gt(0) & out["acceleration"].gt(out["accel_upper"])
        correction = saturated & out["roc"].lt(0) & out["acceleration"].lt(0) & out["acceleration"].lt(out["accel_lower"])
        out["entry"] = (ready & (propagation | correction)).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        active = False
        pending = False
        armed = False
        entry = peak = stop = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if i == 0 or not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                    continue
                # The prior close's entry decision fills at this open.
                entry = peak = opens[i]
                stop = entry - params.trail_atr * atr[i - 1]
                armed = False
                age = 0
                active = True
            if active:
                age += 1
                if not np.isfinite([highs[i], lows[i], closes[i], atr[i]]).all() or atr[i] <= 0:
                    active = False
                    continue
                # Check the existing stop before raising it with today's high.
                hit = lows[i] <= stop
                peak = max(peak, highs[i])
                if not hit:
                    armed = armed or peak >= entry * (1.0 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    # A new stop is breached by the close, never retrospectively by the low.
                    hit = closes[i] <= stop
                if hit or age >= params.hold_bars:
                    active = False
                else:
                    raw[i] = 1
            elif entries[i] == 1 and np.isfinite(closes[i]) and closes[i] > 0:
                raw[i] = 1
                pending = True
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        # All decisions, including stop exits, execute on the next bar.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = Params()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100.0] * 7,
        "high": [100.0, 102.0, 101.0, 101.0, 101.0, 101.0, 101.0],
        "low": [99.5, 99.5, 99.9, 99.5, 99.5, 99.5, 99.5],
        "close": [100.0, 101.0, 100.5, 100.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 7,
    }, index=index)
    indicators = pd.DataFrame({"atr": 1.0, "entry": [1, 0, 0, 0, 0, 0, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    flat = data.copy()
    flat["high"] = 100.5
    flat["low"] = 99.5
    flat["close"] = 100.0
    result = GeneratedStrategy.generate_signals(flat, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    future = data.copy()
    future.iloc[3:, future.columns.get_loc("close")] = 120.0
    changed = GeneratedStrategy.generate_signals(future, indicators, None, params).data
    original = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert changed.iloc[:4].equals(original.iloc[:4])
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
