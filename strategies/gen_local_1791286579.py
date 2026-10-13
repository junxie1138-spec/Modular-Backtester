from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    rank_window: int = 60
    shape_bars: int = 3
    history_bars: int = 252
    neighbors: int = 12
    agreement: float = 0.67
    profit_target: float = 0.008
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791286579"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(params.rank_window + params.history_bars + 2 * params.shape_bars + 3)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        p = params
        if not (p.rank_window >= 2 and p.shape_bars >= 2 and
                1 <= p.neighbors <= p.history_bars and
                0.5 < p.agreement <= 1 and p.profit_target > 0 and
                p.max_hold in (1, 2)):
            raise ValueError("Invalid analog parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["volume_change"] = volume.pct_change(fill_method=None)
        out["two_bar_outcome"] = close.div(data["open"].shift(1).where(data["open"].shift(1) > 0)) - 1.0
        outcomes = out["two_bar_outcome"].to_numpy()
        for primitive in ("return", "volume_change"):
            out[primitive + "_rank"] = out[primitive].rolling(p.rank_window).rank(pct=True)
            columns = []
            for lag in range(p.shape_bars - 1, -1, -1):
                name = primitive + "_rank_lag_" + str(lag)
                out[name] = out[primitive + "_rank"].shift(lag)
                columns.append(name)
            shapes = out[columns].to_numpy()
            forecast = np.full(len(data), np.nan)
            vote = np.full(len(data), np.nan)
            # ponytail: O(bars * history * shape); index neighbors if histories become large.
            for i in range(self.warmup_bars(p) - 1, len(data)):
                if not np.isfinite(shapes[i]).all():
                    continue
                # Separate episodes; every candidate's two-day outcome is already observed.
                end = i - p.shape_bars - 1
                candidates = np.arange(max(p.rank_window + p.shape_bars - 1,
                                           end - p.history_bars), end)
                valid = np.isfinite(shapes[candidates]).all(axis=1) & np.isfinite(outcomes[candidates + 2])
                candidates = candidates[valid]
                if len(candidates) < p.neighbors:
                    continue
                distance = np.square(shapes[candidates] - shapes[i]).mean(axis=1)
                chosen = candidates[np.argsort(distance, kind="stable")[:p.neighbors]]
                realized = outcomes[chosen + 2]
                forecast[i] = realized.mean()
                vote[i] = (realized > 0).mean()
            out[primitive + "_forecast"] = forecast
            out[primitive + "_vote"] = vote
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        long_ok = pd.Series(True, index=data.index)
        short_ok = pd.Series(True, index=data.index)
        for primitive in ("return", "volume_change"):
            forecast = indicators[primitive + "_forecast"]
            vote = indicators[primitive + "_vote"]
            long_ok &= (forecast > 0) & (vote >= params.agreement)
            short_ok &= (forecast < 0) & (vote <= 1.0 - params.agreement)
        entries = np.select([long_ok, short_ok], [1, -1], default=0)
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position, entry_bar, entry_price = 0, -1, np.nan
        for i in range(len(data)):
            if position:
                if entry_bar < 0:
                    entry_bar, entry_price = i, opens[i]
                if not (np.isfinite(entry_price) and entry_price > 0 and
                        np.isfinite(closes[i]) and closes[i] > 0):
                    position = 0
                else:
                    gain = position * (closes[i] / entry_price - 1.0)
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                        position = 0
                # An exit cannot reenter at the same close.
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position, entry_bar = int(entries[i]), -1
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), AnalogParams()
    idx = pd.date_range("2020-01-01", periods=6)
    for direction, closing in ((1, [100, 101, 100, 100, 100, 100]),
                               (-1, [100, 99, 100, 100, 100, 100]),
                               (1, [100] * 6)):
        data = pd.DataFrame({"open": 100.0, "close": closing}, index=idx)
        ind = pd.DataFrame(index=idx)
        for primitive in ("return", "volume_change"):
            ind[primitive + "_forecast"] = float(direction)
            ind[primitive + "_vote"] = 1.0 if direction == 1 else 0.0
        result = strategy.generate_signals(data, ind, None, params).data
        expected = [0, direction, 0] if closing[1] != 100 else [0, direction, direction, 0]
        assert result["signal"].iloc[:len(expected)].tolist() == expected
        ind["volume_change_forecast"] = -float(direction)
        assert not strategy.generate_signals(data, ind, None, params).data["signal"].any()
    t = np.arange(420)
    close = 100 * np.exp(0.0002 * t + 0.01 * np.sin(t / 4))
    data = pd.DataFrame({"open": close * (1 + 0.001 * np.cos(t)),
                         "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": 1000000 * (2 + np.sin(t / 7))},
                        index=pd.date_range("2020-01-01", periods=len(t)))
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:380], params)
    pd.testing.assert_frame_equal(full.iloc[:380], prefix)
    a = strategy.generate_signals(data, full, None, params).data
    b = strategy.generate_signals(data.iloc[:380], prefix, None, params).data
    pd.testing.assert_frame_equal(a.iloc[:380], b)


if __name__ == "__main__":
    _self_check()
