from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PressureParams:
    volume_window: int = 20
    rank_window: int = 60
    atr_window: int = 14
    tail_pct: float = 0.15
    activation_pct: float = 0.0075
    trail_atr: float = 1.5
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[PressureParams]):
    strategy_id = "gen_local-w2_1791313775"

    @classmethod
    def params_type(cls):
        return PressureParams

    def warmup_bars(self, params: PressureParams) -> int:
        return int(max(params.volume_window + params.rank_window, params.atr_window + 1) + 1)

    def indicators(self, data: pd.DataFrame, params: PressureParams) -> pd.DataFrame:
        if min(params.volume_window, params.rank_window, params.atr_window) < 2:
            raise ValueError("Rolling windows must be at least two.")
        if not (0 < params.tail_pct < 0.5 and params.activation_pct > 0 and params.trail_atr > 0):
            raise ValueError("Invalid percentile or stop parameters.")
        if params.max_hold < 1:
            raise ValueError("max_hold must be positive.")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["volume_mean"] = data["volume"].rolling(params.volume_window, min_periods=params.volume_window).mean()
        out["relative_volume"] = data["volume"] / out["volume_mean"].where(out["volume_mean"] > 0)
        out["close_location"] = (2 * data["close"] - data["high"] - data["low"]) / out["range"]
        out["pressure"] = (out["close_location"] * out["relative_volume"]).replace([np.inf, -np.inf], np.nan)
        # API: [pandas Rolling.rank](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.typing.Rolling.rank.html)
        out["percentile"] = out["pressure"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([data["high"] - data["low"], (data["high"] - out["previous_close"]).abs(), (data["low"] - out["previous_close"]).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["pressure_change"] = out["pressure"].diff()
        out["close_change"] = data["close"].diff()
        out["low_tail"] = (out["percentile"] <= params.tail_pct) & (out["pressure"] < 0)
        out["high_tail"] = (out["percentile"] >= 1 - params.tail_pct) & (out["pressure"] > 0)
        out["long_confirmed"] = out["low_tail"] & out["low_tail"].shift(1, fill_value=False) & (out["pressure_change"] > 0) & (out["close_change"] < 0)
        out["short_confirmed"] = out["high_tail"] & out["high_tail"].shift(1, fill_value=False) & (out["pressure_change"] < 0) & (out["close_change"] > 0)
        out["entry_side"] = np.select([out["long_confirmed"], out["short_confirmed"]], [1, -1], default=0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: PressureParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["entry_side"].fillna(0).to_numpy(dtype=int)
        side = 0
        pending = False
        entry = stop = best = 0.0
        armed = False
        held = 0
        for i, (opening, high, low, close) in enumerate(prices):
            if not (np.isfinite(prices[i]).all() and np.all(prices[i] > 0) and np.isfinite(atr[i]) and atr[i] > 0):
                side = 0
                pending = False
                continue
            if pending:
                # Yesterday's close decision fills at today's open.
                entry = side * opening
                best = entry
                stop = entry - params.trail_atr * atr[i - 1]
                held = 0
                armed = False
                pending = False
            if side:
                held += 1
                adverse = low if side == 1 else -high
                favorable = high if side == 1 else -low
                # Test the previously established stop before updating it.
                # A touch requests an exit at the following open, not a stop-price fill.
                if adverse <= stop or held >= params.max_hold:
                    side = 0
                else:
                    best = max(best, favorable)
                    armed = armed or best - entry >= abs(entry) * params.activation_pct
                    if armed:
                        # In side-adjusted coordinates the stop only moves up.
                        # This raises long stops and lowers short buyback stops.
                        stop = max(stop, entry, best - params.trail_atr * atr[i])
                raw[i] = side
                continue
            if i >= self.warmup_bars(params) and candidates[i]:
                side = int(candidates[i])
                pending = True
                raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = PressureParams(volume_window=2, rank_window=2, atr_window=2, activation_pct=0.01, trail_atr=1.0, max_hold=4)
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=12)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    for direction in (1, -1):
        rows = [[100.0, 100.5, 99.5, 100.0, 1000.0] for _ in index]
        rows[6] = [100.0, 102.5, 99.5, 102.0, 1000.0] if direction == 1 else [100.0, 100.5, 97.5, 98.0, 1000.0]
        rows[7] = [102.0, 102.2, 101.0, 101.2, 1000.0] if direction == 1 else [98.0, 99.0, 97.8, 98.8, 1000.0]
        data = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=index)
        derived = pd.DataFrame({"atr": 1.0, "entry_side": 0}, index=index)
        derived.iloc[5, derived.columns.get_loc("entry_side")] = direction
        result = strategy.generate_signals(data, derived, ctx, params).data
        assert result["signal"].tolist() == [0] * 6 + [direction, direction] + [0] * 4
        assert result["size"].gt(0).all()
        data.loc[index[6]:, "high"] = 100.5
        data.loc[index[6]:, "low"] = 99.5
        derived["atr"] = 10.0
        result = strategy.generate_signals(data, derived, ctx, params).data
        assert result["signal"].tolist() == [0] * 6 + [direction] * 4 + [0] * 2
    flat = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    assert not strategy.indicators(flat, params)["entry_side"].any()


if __name__ == "__main__":
    _self_check()
