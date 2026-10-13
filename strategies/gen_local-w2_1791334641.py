from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TailRankParams:
    tail_window: int = 5
    rank_window: int = 63
    rank_threshold: float = 0.80
    atr_window: int = 14
    breakeven_pct: float = 0.01
    trail_atr: float = 1.5
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[TailRankParams]):
    strategy_id = "gen_local-w2_1791334641"

    @classmethod
    def params_type(cls):
        return TailRankParams

    def warmup_bars(self, params: TailRankParams) -> int:
        return int(max(params.tail_window + params.rank_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: TailRankParams) -> pd.DataFrame:
        if min(params.tail_window, params.rank_window, params.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 0.5 < params.rank_threshold < 1.0:
            raise ValueError("rank_threshold must be between 0.5 and 1.")
        if not 3 <= params.max_hold <= 5:
            raise ValueError("max_hold must be between three and five.")
        if not np.isfinite(params.breakeven_pct) or params.breakeven_pct <= 0:
            raise ValueError("breakeven_pct must be positive and finite.")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        previous = close.shift(1)
        out["return"] = close.div(previous).sub(1).replace([np.inf, -np.inf], np.nan)
        out["weakest_return"] = out["return"].rolling(params.tail_window).min()
        out["strongest_return"] = out["return"].rolling(params.tail_window).max()
        out["floor_rank"] = out["weakest_return"].rolling(params.rank_window).rank(pct=True)
        out["ceiling_rank"] = out["strongest_return"].rolling(params.rank_window).rank(pct=True)
        out["window_return"] = close.div(close.shift(params.tail_window)).sub(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        ready = out[["floor_rank", "ceiling_rank", "window_return", "atr"]].notna().all(axis=1)
        ready &= out["atr"].gt(0)
        out["long_candidate"] = ready & out["floor_rank"].ge(params.rank_threshold) & out["window_return"].gt(0)
        out["short_candidate"] = ready & out["ceiling_rank"].le(1 - params.rank_threshold) & out["window_return"].lt(0)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: TailRankParams
    ) -> SignalFrame:
        long_ok = indicators["long_candidate"].fillna(False).astype(bool)
        short_ok = indicators["short_candidate"].fillna(False).astype(bool)
        long_entry = (long_ok & long_ok.shift(1, fill_value=False)).to_numpy()
        short_entry = (short_ok & short_ok.shift(1, fill_value=False)).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        pending = False
        armed = False
        entry = stop = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if side:
                if pending:
                    # The preceding close requested a fill at this session's open.
                    entry = opens[i]
                    entry_bar = i
                    pending = False
                    if not np.isfinite(entry) or entry <= 0:
                        side = 0
                        continue
                    stop = entry - side * params.trail_atr * atrs[i - 1]
                close = closes[i]
                atr = atrs[i]
                if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
                    side = 0
                    continue
                # Stops are evaluated at closes; the exit fills next session.
                if side * (close - stop) <= 0 or i - entry_bar + 1 >= params.max_hold:
                    side = 0
                    continue
                if side * (close / entry - 1) >= params.breakeven_pct:
                    armed = True
                if armed:
                    candidate = close - side * params.trail_atr * atr
                    # Long stops only rise; short stops only fall.
                    stop = max(stop, entry, candidate) if side == 1 else min(stop, entry, candidate)
                raw[i] = side
            elif long_entry[i] or short_entry[i]:
                side = 1 if long_entry[i] else -1
                pending = True
                armed = False
                raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = TailRankParams(trail_atr=1.0)
    index = pd.date_range("2020-01-01", periods=11)
    close = pd.Series([100, 100, 101, 103, 102, 100, 99, 98, 96, 97, 99], index=index, dtype=float)
    data = pd.DataFrame({"open": close.shift(1).fillna(close.iloc[0]),
                         "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1000.0})
    ind = pd.DataFrame({"atr": 1.0, "long_candidate": False,
                        "short_candidate": False}, index=index)
    ind.loc[index[:2], "long_candidate"] = True
    ind.loc[index[6:8], "short_candidate"] = True
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    flat = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0},
                        index=pd.date_range("2020-01-01", periods=100))
    derived = strategy.indicators(flat, params)
    assert strategy.generate_signals(flat, derived, ctx, params).data["signal"].eq(0).all()
    assert strategy.warmup_bars(params) >= params.tail_window + params.rank_window + 1


if __name__ == "__main__":
    _self_check()
