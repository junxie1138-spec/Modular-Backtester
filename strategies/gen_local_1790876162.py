from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    energy_window: int = 12
    coarse_share: float = 0.50
    atr_window: int = 14
    breakeven_pct: float = 0.01
    trail_atr: float = 1.5
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790876162"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        if not (params.energy_window >= 2 and params.atr_window >= 2
                and 0.0 < params.coarse_share < 1.0
                and params.breakeven_pct > 0.0 and params.trail_atr > 0.0
                and 3 <= params.hold_days <= 5):
            raise ValueError("Invalid strategy parameters")
        return int(max(params.energy_window + 9, params.atr_window + 2))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1).where(lambda s: s > 0.0)
        out["gap"] = data["open"].div(previous).sub(1.0)
        out["intraday"] = data["close"].div(data["open"].where(data["open"] > 0.0)).sub(1.0)
        out["gap_two"] = out["gap"].rolling(2).mean()
        out["gap_four"] = out["gap"].rolling(4).mean()
        # Causal undecimated Haar details at two, four and eight sessions.
        out["detail_two"] = (out["gap"] - out["gap"].shift(1)) / np.sqrt(2.0)
        out["detail_four"] = out["gap_two"] - out["gap_two"].shift(2)
        out["detail_eight"] = np.sqrt(2.0) * (out["gap_four"] - out["gap_four"].shift(4))
        for scale in ("two", "four", "eight"):
            out["energy_" + scale] = out["detail_" + scale].pow(2).rolling(params.energy_window).mean()
        total = out[["energy_two", "energy_four", "energy_eight"]].sum(axis=1, min_count=3)
        out["coarse_fraction"] = out["energy_eight"].div(total.where(total > 0.0))
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1) & (out["atr"] > 0.0)
        coherent = ready & (out["coarse_fraction"] >= params.coarse_share)
        long = coherent & (out["gap_four"] > 0.0) & (out["gap_two"] > 0.0) & (out["intraday"] > 0.0)
        short = coherent & (out["gap_four"] < 0.0) & (out["gap_two"] < 0.0) & (out["intraday"] < 0.0)
        out["candidate"] = np.select([long, short], [1, -1], default=0).astype(int)
        out["confirmed"] = out["candidate"].where(
            out["candidate"].eq(out["candidate"].shift(1)), 0
        ).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        self.warmup_bars(params)
        decisions = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        votes = indicators["confirmed"].fillna(0).to_numpy(dtype=int)
        side, age = 0, 0
        entry, stop, entry_atr = np.nan, 0.0, 0.0
        armed = False
        for i in range(len(data)):
            if side:
                # Entry price becomes observable on the following bar only.
                if np.isnan(entry):
                    entry = opens[i]
                    if not np.isfinite(entry) or entry <= 0.0:
                        side = 0
                        continue
                    stop = -params.trail_atr * entry_atr
                age += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0.0:
                    side = 0
                    continue
                profit = side * (closes[i] - entry)
                # Stops use closes; an exit decision fills at the next open.
                if profit <= stop or age >= params.hold_days:
                    side = 0
                    entry = np.nan
                    continue
                if profit / entry >= params.breakeven_pct:
                    armed = True
                if armed:
                    # Signed profit coordinates ratchet upward for either side.
                    stop = max(stop, 0.0)
                    if np.isfinite(atrs[i]) and atrs[i] > 0.0:
                        stop = max(stop, profit - params.trail_atr * atrs[i])
                decisions[i] = side
            elif votes[i] in (-1, 1) and np.isfinite(atrs[i]) and atrs[i] > 0.0:
                side = int(votes[i])
                entry, entry_atr = np.nan, atrs[i]
                age, armed = 0, False
                decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams(trail_atr=1.0)
    index = pd.date_range("2020-01-01", periods=7)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for side in (1, -1):
        for movement, expected in (
            ([0, 0, 0, 2, 0.5, 0, 0], [0, 0, side, side, side, 0, 0]),
            ([0, 0, 0, 0, 0, 0, 0], [0, 0, side, side, side, side, 0]),
        ):
            close = 100.0 + side * np.array(movement)
            data = pd.DataFrame({
                "open": 100.0, "high": np.maximum(close, 100.0) + 1.0,
                "low": np.minimum(close, 100.0) - 1.0, "close": close, "volume": 1000.0,
            }, index=index)
            ind = pd.DataFrame({"atr": 1.0, "confirmed": [0, side, 0, 0, 0, 0, 0]}, index=index)
            result = strategy.generate_signals(data, ind, ctx, params).data
            assert result["signal"].tolist() == expected
            assert result["size"].gt(0.0).all()
    assert strategy.indicators(data, params)["confirmed"].eq(0).all()
    t = np.arange(100, dtype=float)
    close = 100.0 + 2.0 * np.sin(t / 4.0)
    opening = np.r_[100.0, close[:-1]] * (1.0 + 0.003 * np.sin(t / 4.0))
    data = pd.DataFrame({
        "open": opening, "high": np.maximum(opening, close) + 0.5,
        "low": np.minimum(opening, close) - 0.5, "close": close, "volume": 1000.0,
    }, index=pd.date_range("2020-01-01", periods=100))
    ind = strategy.indicators(data, params)
    assert ind["confirmed"].ne(0).any()
    active = ind["confirmed"].ne(0)
    assert ind.loc[active, "candidate"].eq(ind["candidate"].shift(1).loc[active]).all()
    prefix = strategy.indicators(data.iloc[:70], params)
    pd.testing.assert_frame_equal(prefix, ind.iloc[:70])
    full = strategy.generate_signals(data, ind, ctx, params).data
    partial = strategy.generate_signals(data.iloc[:70], prefix, ctx, params).data
    pd.testing.assert_frame_equal(partial, full.iloc[:70])
