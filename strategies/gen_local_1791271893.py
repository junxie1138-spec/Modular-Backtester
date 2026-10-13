from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    complexity_window: int = 30
    complexity_gap: float = 0.20
    coarse_entropy_max: float = 0.60
    breakeven_pct: float = 0.025
    initial_atr: float = 2.5
    trail_atr: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791271893"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(200, params.complexity_window + 10, 15)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.complexity_window < 2 or params.holding_bars < 1:
            raise ValueError("Window must be >= 2 and holding_bars >= 1")
        if not (0 <= params.complexity_gap <= 1 and 0 <= params.coarse_entropy_max <= 1):
            raise ValueError("Complexity thresholds must be in [0, 1]")
        if not (params.breakeven_pct > 0 and params.initial_atr > 0 and params.trail_atr > 0):
            raise ValueError("Stop parameters must be positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        for name, lag in (("fine", 1), ("coarse", 5)):
            a, b, c = close.shift(2 * lag), close.shift(lag), close
            valid = a.notna() & b.notna() & c.notna() & a.ne(b) & a.ne(c) & b.ne(c)
            pattern = (4 * a.gt(b).astype(int) + 2 * a.gt(c).astype(int) + b.gt(c).astype(int)).where(valid)
            out[name + "_pattern"] = pattern
            entropy = pd.Series(0.0, index=data.index)
            for code in (0, 1, 3, 4, 6, 7):
                probability = pattern.eq(code).astype(float).where(valid).rolling(
                    params.complexity_window, min_periods=params.complexity_window
                ).mean()
                out[name + "_p" + str(code)] = probability
                entropy = entropy - probability * np.log(probability.clip(lower=1e-12))
            out[name + "_entropy"] = entropy / np.log(6.0)
        out["complexity_spread"] = out["fine_entropy"] - out["coarse_entropy"]
        out["coarse_return"] = close.pct_change(10, fill_method=None)
        out["entry_ready"] = (
            close.gt(out["ma200"])
            & out["coarse_return"].gt(0)
            & out["coarse_entropy"].le(params.coarse_entropy_max)
            & out["complexity_spread"].ge(params.complexity_gap)
            & out["atr"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.95}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        active, armed = False, False
        entry, stop, peak = np.nan, np.nan, np.nan
        entry_bar = -1
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Yesterday's decision fills at today's open.
                    entry = opens[i]
                    stop = entry - params.initial_atr * atrs[i - 1]
                    peak = entry
                close, atr = closes[i], atrs[i]
                if not (np.isfinite(entry) and entry > 0 and np.isfinite(close)
                        and np.isfinite(atr) and atr > 0):
                    active = False
                else:
                    # Stops are evaluated on closes; all exits fill next open.
                    peak = max(peak, close)
                    armed = armed or close >= entry * (1.0 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr)
                    if close <= stop or i - entry_bar + 1 >= params.holding_bars:
                        active = False
            elif i >= self.warmup_bars(params) - 1 and ready[i]:
                active, armed = True, False
                entry_bar = i + 1
            signals[i] = int(active)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=240, freq="B")
    close = np.full(240, 100.0)
    close[200:205] = [101.0, 103.0, 105.0, 104.0, 101.0]
    data = pd.DataFrame(
        {"open": 100.0, "high": close + 1.0, "low": close - 1.0,
         "close": close, "volume": 1000.0}, index=index
    )
    strategy, params = GeneratedStrategy(), Params()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived["entry_ready"].iloc[:199].any()
    indicators = pd.DataFrame({"atr": 1.0, "entry_ready": False}, index=index)
    indicators.iloc[199, 1] = True
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[199:206].tolist() == [0, 1, 1, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    params.holding_bars = 3
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[202] == 1 and result["signal"].iloc[203] == 0


if __name__ == "__main__":
    _self_check()
