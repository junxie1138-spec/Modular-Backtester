from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rank_window: int = 80
    recent_window: int = 4
    baseline_window: int = 16
    rank_jump: float = 0.22
    atr_window: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791254764"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.rank_window + params.recent_window + params.baseline_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        p = params
        if min(p.rank_window, p.recent_window, p.baseline_window, p.atr_window, p.max_hold) < 1:
            raise ValueError("Window lengths and max_hold must be positive.")
        if not (0.0 < p.rank_jump < 0.5 and p.breakeven_pct > 0.0 and p.trail_atr > 0.0):
            raise ValueError("Invalid rank or stop parameters.")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_rank"] = out["return"].rolling(p.rank_window, min_periods=p.rank_window).rank(pct=True)
        out["recent_rank"] = out["return_rank"].rolling(p.recent_window, min_periods=p.recent_window).mean()
        out["baseline_rank"] = out["return_rank"].shift(p.recent_window).rolling(p.baseline_window, min_periods=p.baseline_window).mean()
        out["rank_shift"] = out["recent_rank"] - out["baseline_rank"]
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_window, min_periods=p.atr_window).mean()
        ready = np.isfinite(out[["recent_rank", "baseline_rank", "rank_shift", "atr"]]).all(axis=1) & out["atr"].gt(0)
        neutral = out["baseline_rank"].between(0.35, 0.65)
        bullish = ready & neutral & out["rank_shift"].ge(p.rank_jump) & out["recent_rank"].ge(0.5 + p.rank_jump / 2)
        bearish = ready & neutral & out["rank_shift"].le(-p.rank_jump) & out["recent_rank"].le(0.5 - p.rank_jump / 2)
        out["vote"] = np.select([bullish, bearish], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        vote = indicators["vote"].fillna(0).astype(int)
        confirmed = vote.ne(0) & vote.eq(vote.shift(1))
        entry_event = confirmed & ~confirmed.shift(1, fill_value=False)
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        votes = vote.to_numpy()
        events = entry_event.to_numpy()
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry = np.nan
        stop = -np.inf
        armed = False
        entry_bar = -1
        for i in range(len(data)):
            if position:
                if np.isnan(entry):
                    # The previous close's instruction fills at this bar's open.
                    entry = opens[i]
                    entry_bar = i
                    if not np.isfinite(entry) or entry <= 0:
                        position = 0
                        continue
                    stop = position * entry - params.trail_atr * atr[i - 1]
                if not np.isfinite(close[i]) or not np.isfinite(atr[i]):
                    position = 0
                    entry = np.nan
                    continue
                signed_close = position * close[i]
                signed_entry = position * entry
                # Stops are evaluated on closes; exits fill at the following open.
                if signed_close <= stop or i - entry_bar + 1 >= params.max_hold:
                    position = 0
                    entry = np.nan
                    continue
                if signed_close - signed_entry >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    # Favorable-coordinate stop only rises; a short's price stop only falls.
                    stop = max(stop, signed_entry, signed_close - params.trail_atr * atr[i])
            elif events[i] and np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(votes[i])
                entry = np.nan
                armed = False
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    p = Params()
    index = pd.date_range("2020-01-01", periods=6)
    for direction in (1, -1):
        closes = np.array([100, 100, 102, 103, 100, 100], dtype=float)
        if direction == -1:
            closes = 200 - closes
        data = pd.DataFrame({"open": 100.0, "high": closes + 1, "low": closes - 1, "close": closes, "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"vote": direction, "atr": 1.0}, index=index)
        result = GeneratedStrategy.generate_signals(None, data, ind, None, p).data
        assert result["signal"].tolist() == [0, 0, direction, direction, direction, 0]
        assert result["size"].gt(0).all()
        ind["vote"] = [direction, 0, direction, 0, direction, 0]
        assert GeneratedStrategy.generate_signals(None, data, ind, None, p).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
