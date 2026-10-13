from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    history_bars: int = 756
    energy_window: int = 16
    atr_bars: int = 14
    holding_bars: int = 7
    min_observations: int = 24
    coarse_cut: float = 0.45
    edge_threshold: float = 0.8
    breakeven_pct: float = 0.012
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791306992"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.atr_bars + 1, 8 + params.energy_window + params.holding_bars + params.history_bars))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        returns = np.log(close).diff()
        out["log_return"] = returns
        for scale in (2, 4, 8):
            half = scale // 2
            average = returns.rolling(half, min_periods=half).mean()
            detail = np.sqrt(scale) * (average - average.shift(half)) / 2.0
            out[f"haar_{scale}"] = detail
            out[f"energy_{scale}"] = detail.pow(2).rolling(p.energy_window, min_periods=p.energy_window).mean()
        energy = out[["energy_2", "energy_4", "energy_8"]].sum(axis=1, min_count=3)
        out["total_energy"] = energy
        out["coarse_share"] = out["energy_8"] / energy.where(energy > 0)
        day = pd.Series(data.index.day, index=data.index)
        out["calendar_phase"] = np.select(
            [(day <= 5) | (day >= 25), day <= 15], [0, 1], default=2
        ).astype(int)
        valid = out["coarse_share"].notna()
        out["energy_regime"] = (out["coarse_share"] >= p.coarse_cut).astype(float).where(valid)
        out["state"] = 2 * out["calendar_phase"] + out["energy_regime"]
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_bars, min_periods=p.atr_bars).mean()
        # At t, this label belongs to the setup at t-h and its next-open entry.
        entry_open = data["open"].shift(p.holding_bars - 1)
        out["matured_return"] = np.log(close / entry_open.where(entry_open > 0))
        out["matured_state"] = out["state"].shift(p.holding_bars)
        out["edge"] = 0.0
        for state in range(6):
            observed = out["matured_state"].eq(state) & out["matured_return"].notna()
            values = out["matured_return"].where(observed, 0.0)
            count = observed.astype(float).rolling(p.history_bars, min_periods=p.history_bars).sum()
            total = values.rolling(p.history_bars, min_periods=p.history_bars).sum()
            squares = values.pow(2).rolling(p.history_bars, min_periods=p.history_bars).sum()
            denominator = count.where(count >= max(2, p.min_observations))
            mean = total / denominator
            variance = ((squares - total.pow(2) / denominator) / (denominator - 1)).clip(lower=0)
            score = mean / np.sqrt(variance.where(variance > 1e-12) / denominator)
            out[f"count_{state}"] = count
            out[f"mean_{state}"] = mean
            out[f"variance_{state}"] = variance
            out[f"score_{state}"] = score
            current = out["state"].eq(state)
            out.loc[current, "edge"] = score.loc[current]
        out["edge"] = out["edge"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        p = params
        edge = indicators["edge"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decision = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.15, dtype=float)
        side = 0
        entry = stop = best = 0.0
        armed = False
        held = 0
        position_size = 0.15
        warmup = self.warmup_bars(p)
        for i in range(len(data)):
            # A decision on i-1 establishes the position at open[i].
            if side == 0 and i > 0 and decision[i - 1] != 0:
                if not (np.isfinite(opens[i]) and opens[i] > 0 and
                        np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    continue
                side = int(decision[i - 1])
                entry = opens[i]
                best = side * entry
                stop = best - p.trail_atr * atr[i - 1]
                armed = False
                held = 0
                position_size = sizes[i - 1]
            if side != 0:
                held += 1
                if not (np.isfinite(closes[i]) and closes[i] > 0 and
                        np.isfinite(atr[i]) and atr[i] > 0):
                    side = 0
                    continue
                # Signed prices make the stop move only upward for either side.
                # Stops use closing prices; exits execute at the next open.
                signed_close = side * closes[i]
                exit_now = signed_close <= stop or held >= p.holding_bars
                if not exit_now:
                    best = max(best, signed_close)
                    armed = armed or best >= side * entry + entry * p.breakeven_pct
                    if armed:
                        stop = max(stop, side * entry, best - p.trail_atr * atr[i])
                    exit_now = signed_close <= stop
                if exit_now:
                    side = 0
                    continue
                decision[i] = side
                sizes[i] = position_size
            elif (i >= warmup and np.isfinite(edge[i]) and
                  abs(edge[i]) >= p.edge_threshold and np.isfinite(atr[i]) and atr[i] > 0):
                decision[i] = 1 if edge[i] > 0 else -1
                sizes[i] = 0.15 + 0.80 * np.clip((abs(edge[i]) - p.edge_threshold) / 2.0, 0.0, 1.0)
        df = pd.DataFrame({"signal": decision, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.15).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    p = Params(history_bars=4, energy_window=2, min_observations=2)
    index = pd.date_range("2020-01-01", periods=60, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 105.0, "low": 95.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[[26, 27, 28, 33, 34, 35]], "close"] = [102, 104, 101, 98, 96, 99]
    indicators = pd.DataFrame({"edge": 0.0, "atr": 1.0}, index=index)
    indicators.loc[index[25], "edge"] = 2.0
    indicators.loc[index[32], "edge"] = -2.0
    result = strategy.generate_signals(data, indicators, None, p).data
    assert result["signal"].iloc[25] == 0 and result["signal"].iloc[26] == 1
    assert result["signal"].iloc[28] == 1 and result["signal"].iloc[29] == 0
    assert result["signal"].iloc[32] == 0 and result["signal"].iloc[33] == -1
    assert result["signal"].iloc[35] == -1 and result["signal"].iloc[36] == 0
    assert result["size"].between(0.15, 0.95).all()
    calculated = strategy.indicators(data, p)
    prefix = strategy.indicators(data.iloc[:40], p)
    pd.testing.assert_frame_equal(calculated.iloc[:40], prefix)
    assert np.isfinite(calculated["edge"]).all()


if __name__ == "__main__":
    _self_check()
