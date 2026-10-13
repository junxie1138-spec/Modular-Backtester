from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalReleaseParams:
    fast_window: int = 6
    reference_window: int = 24
    complexity_drop: float = 0.02
    entropy_rise: float = 0.05
    atr_window: int = 14
    breakeven_pct: float = 0.004
    trail_atr: float = 0.8
    hold_bars: int = 2
    min_size: float = 0.2
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[OrdinalReleaseParams]):
    strategy_id = "gen_local-w2_1790963365"

    @classmethod
    def params_type(cls):
        return OrdinalReleaseParams

    def warmup_bars(self, params: OrdinalReleaseParams) -> int:
        return max(params.fast_window + params.reference_window + 3,
                   params.atr_window + 1, 4)

    def indicators(self, data: pd.DataFrame, params: OrdinalReleaseParams) -> pd.DataFrame:
        if (params.fast_window < 3 or params.reference_window < 3
                or params.atr_window < 1 or params.complexity_drop <= 0
                or not 0 <= params.entropy_rise <= 1
                or params.breakeven_pct <= 0 or params.trail_atr <= 0
                or params.hold_bars not in (1, 2)
                or not 0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Invalid ordinal-release parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["three_bar_return"] = close / close.shift(3) - 1.0
        triples = pd.concat([out["return"].shift(2),
                             out["return"].shift(1), out["return"]], axis=1)
        valid = np.isfinite(triples).all(axis=1)
        # Stable sorting assigns equal returns a deterministic time-order rank.
        order = np.argsort(triples.to_numpy(), axis=1, kind="stable")
        code = pd.Series(order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2],
                         index=data.index, dtype=float).where(valid)
        out["ordinal_code"] = code
        codes = (5, 7, 11, 15, 19, 21)
        for label, window, lag in (
                ("fast", params.fast_window, 0),
                ("reference", params.reference_window, params.fast_window)):
            probabilities = pd.DataFrame(index=data.index)
            for number in codes:
                occurrence = (code == number).astype(float).where(valid)
                probabilities[number] = occurrence.rolling(
                    window, min_periods=window).mean().shift(lag)
                out[f"{label}_p_{number}"] = probabilities[number]
            complete = probabilities.notna().all(axis=1)
            entropy = -(probabilities * np.log(
                probabilities.clip(lower=1e-12))).sum(axis=1) / np.log(6.0)
            concentration = ((probabilities - 1.0 / 6.0) ** 2).sum(axis=1) * 6.0 / 5.0
            out[f"{label}_entropy"] = entropy.where(complete)
            out[f"{label}_complexity"] = (entropy * concentration).where(complete)
        out["complexity_release"] = out["reference_complexity"] - out["fast_complexity"]
        out["entropy_release"] = out["fast_entropy"] - out["reference_entropy"]
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out["strength"] = (out["complexity_release"] /
                           (3.0 * params.complexity_drop)).clip(0.0, 1.0)
        out["entry"] = (
            (out["complexity_release"] >= params.complexity_drop)
            & (out["entropy_release"] >= params.entropy_rise)
            & (out["three_bar_return"] < 0)
            & (out["return"] > 0)
            & (out["atr"] > 0)).fillna(False)
        out.iloc[:self.warmup_bars(params), out.columns.get_loc("entry")] = False
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: OrdinalReleaseParams) -> SignalFrame:
        count = len(data)
        decisions = np.zeros(count, dtype=int)
        sizes = np.full(count, params.min_size, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        strength = indicators["strength"].fillna(0.0).to_numpy(dtype=float)
        active = armed = False
        age = 0
        entry = peak = stop = 0.0
        held_size = params.min_size
        for i in range(count):
            # An entry decision from yesterday becomes today's opening position.
            if not active and i > 0 and decisions[i - 1] == 1:
                if not (np.isfinite(opens[i]) and opens[i] > 0
                        and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    continue
                active = True
                armed = False
                age = 0
                entry = peak = opens[i]
                stop = entry - params.trail_atr * atr[i - 1]
                held_size = sizes[i - 1]
            if active:
                age += 1
                if not (np.isfinite(highs[i]) and np.isfinite(lows[i])
                        and np.isfinite(closes[i]) and np.isfinite(atr[i])
                        and closes[i] > 0 and atr[i] > 0):
                    active = False
                    continue
                hit_old_stop = lows[i] <= stop
                peak = max(peak, highs[i])
                if peak >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                # New intraday highs tighten the close-time stop; do not assume
                # whether today's low occurred before or after today's high.
                exit_now = (hit_old_stop or (armed and closes[i] <= stop)
                            or age >= params.hold_bars)
                if exit_now:
                    active = False
                    continue
                decisions[i] = 1
                sizes[i] = held_size
            elif entries[i]:
                decisions[i] = 1
                sizes[i] = (params.min_size + strength[i]
                            * (params.max_size - params.min_size))
        df = pd.DataFrame({"signal": decisions, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = OrdinalReleaseParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0] * 6,
        "high": [100.1, 100.1, 101.0, 100.4, 100.1, 100.1],
        "low": [99.9, 99.9, 99.9, 100.1, 99.9, 99.9],
        "close": [100.0, 100.0, 100.9, 100.2, 100.0, 100.0],
        "volume": [1000.0] * 6}, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0] * 6,
        "entry": [False, True, False, False, False, False],
        "strength": [0.0, 0.5, 0.0, 0.0, 0.0, 0.0]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    assert np.isclose(result["size"].iloc[2], 0.6)
    assert (result["size"] > 0).all()
    params.hold_bars = 1
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, 0]
    params.hold_bars = 2
    params.trail_atr = 0.2
    data.loc[index[2], "close"] = 100.7
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, 0]
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived["entry"].any()


if __name__ == "__main__":
    _self_check()
