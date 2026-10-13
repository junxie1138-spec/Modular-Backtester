from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 20
    selloff_window: int = 5
    selloff_z: float = 0.8
    continuation_deficit: float = 0.10
    atr_period: int = 14
    breakeven_pct: float = 0.004
    trail_atr: float = 1.0
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791294264"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.lookback + 3, params.selloff_window + 2, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.lookback < 3 or params.selloff_window < 1 or params.atr_period < 1
                or params.max_hold not in (1, 2) or params.selloff_z <= 0
                or not 0 < params.continuation_deficit < 1
                or params.breakeven_pct <= 0 or params.trail_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["down"] = out["return"].lt(0).astype(float).where(out["return"].notna())
        out["previous_down"] = out["down"].shift(1)
        out["down_pair"] = out["down"] * out["previous_down"]
        n = params.lookback
        out["down_frequency"] = out["down"].rolling(n, min_periods=n).mean()
        out["down_opportunities"] = out["previous_down"].rolling(n, min_periods=n).sum()
        out["down_pairs"] = out["down_pair"].rolling(n, min_periods=n).sum()
        out["continuation"] = out["down_pairs"] / out["down_opportunities"].replace(0, np.nan)
        out["continuation_gap"] = out["continuation"] - out["down_frequency"]
        out["return_vol"] = out["return"].rolling(n, min_periods=n).std(ddof=1)
        m = params.selloff_window
        out["net_return"] = out["return"].rolling(m, min_periods=m).sum()
        out["selloff_score"] = out["net_return"] / (out["return_vol"].replace(0, np.nan) * np.sqrt(m))
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        out["eligible"] = (
            (out["selloff_score"] <= -params.selloff_z)
            & (out["continuation_gap"] <= -params.continuation_deficit)
            & np.isfinite(out["atr"]) & (out["atr"] > 0)
        ).fillna(False)
        out["confirmed"] = out["eligible"] & out["eligible"].shift(1, fill_value=False)
        out["entry_event"] = out["confirmed"] & ~out["confirmed"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        bars = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_event"].fillna(False).to_numpy(dtype=bool)
        pending = False
        active = False
        armed = False
        entry = peak = stop = 0.0
        age = 0
        for i, (opening, high, low, close) in enumerate(bars):
            valid = np.isfinite([opening, high, low, close]).all() and min(opening, low, close) > 0
            if pending:
                pending = False
                if valid and i > 0 and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0:
                    entry = opening
                    peak = opening
                    stop = entry - params.trail_atr * atrs[i - 1]
                    age = 0
                    armed = False
                    active = True
            if active:
                age += 1
                # Test the previously established stop before raising it at this close.
                exit_now = not valid or low <= stop
                if not exit_now:
                    peak = max(peak, high)
                    armed = armed or peak >= entry * (1.0 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop = max(stop, peak - params.trail_atr * atrs[i])
                        exit_now = close <= stop
                if exit_now or age >= params.max_hold:
                    active = False
                else:
                    signals[i] = 1
            elif valid and entries[i]:
                signals[i] = 1
                pending = True
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    eligible = pd.Series([False, True, True, False, False, False, False], index=index)
    confirmed = eligible & eligible.shift(1, fill_value=False)
    ind = pd.DataFrame({"atr": 1.0, "entry_event": confirmed & ~confirmed.shift(1, fill_value=False)}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    for high, close, expected in (
        (100.1, 100.0, [0, 0, 0, 1, 1, 0, 0]),
        (100.8, 99.95, [0, 0, 0, 1, 0, 0, 0]),
        (102.0, 100.5, [0, 0, 0, 1, 0, 0, 0]),
    ):
        sample = data.copy()
        sample.loc[index[3], ["high", "low", "close"]] = [high, 99.5, close]
        result = strategy.generate_signals(sample, ind, ctx, params).data
        assert result["signal"].tolist() == expected
        assert result["size"].gt(0).all()
    warmup = strategy.indicators(data, params)
    assert not warmup["entry_event"].any()


if __name__ == "__main__":
    _self_check()
