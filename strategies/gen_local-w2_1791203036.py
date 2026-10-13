from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    pattern_window: int = 20
    shape_change: int = 5
    analog_window: int = 252
    neighbors: int = 12
    percentile_window: int = 126
    entry_quantile: float = 0.75
    hold_bars: int = 8
    atr_window: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791203036"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(
            params.atr_window + 1,
            3 + params.pattern_window + params.shape_change
            + params.hold_bars + params.analog_window
            + params.percentile_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        windows = (params.pattern_window, params.shape_change,
                   params.analog_window, params.percentile_window,
                   params.hold_bars, params.atr_window)
        if (min(windows) < 1 or not 1 <= params.neighbors <= params.analog_window
                or not 0 < params.entry_quantile < 1
                or params.breakeven_pct <= 0 or params.trail_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        triples = np.column_stack([
            out["return"].shift(2), out["return"].shift(1), out["return"]])
        order = np.argsort(triples, axis=1, kind="stable")
        codes = order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2]
        valid = np.isfinite(triples).all(axis=1)
        out["pattern"] = np.where(valid, codes, np.nan)
        probability_columns = []
        for code in (5, 7, 11, 15, 19, 21):
            name = f"p_{code}"
            probability_columns.append(name)
            out[name] = out["pattern"].eq(code).astype(float).where(
                out["pattern"].notna()).rolling(params.pattern_window).mean()
        probabilities = out[probability_columns]
        out["entropy"] = -(probabilities * np.log(
            probabilities.where(probabilities > 0, 1.0))).sum(
                axis=1, min_count=6) / np.log(6.0)
        out["entropy_change"] = out["entropy"].diff(params.shape_change)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        # At t this is the realized label of the shape observed at t-hold_bars:
        # next-open entry followed by a close hold_bars sessions later.
        out["matured_return"] = (
            data["close"] / data["open"].shift(params.hold_bars - 1) - 1
        ).replace([np.inf, -np.inf], np.nan)
        features = out[probability_columns + ["entropy", "entropy_change"]]
        current = features.to_numpy()
        distances = np.full((len(data), params.analog_window), np.inf)
        outcomes = np.full_like(distances, np.nan)
        for column in range(params.analog_window):
            lag = params.hold_bars + 1 + column
            past = features.shift(lag).to_numpy()
            outcome = out["matured_return"].shift(
                lag - params.hold_bars).to_numpy()
            distance = np.abs(current - past).sum(axis=1)
            usable = np.isfinite(distance) & np.isfinite(outcome)
            distances[:, column] = np.where(usable, distance, np.inf)
            outcomes[:, column] = outcome
        nearest = np.argsort(distances, axis=1, kind="stable")[:, :params.neighbors]
        chosen_distances = np.take_along_axis(distances, nearest, axis=1)
        chosen_outcomes = np.take_along_axis(outcomes, nearest, axis=1)
        complete = np.isfinite(chosen_distances).all(axis=1)
        out["analog_distance"] = np.where(
            complete, chosen_distances.mean(axis=1), np.nan)
        out["forecast"] = np.where(
            complete, chosen_outcomes.mean(axis=1), np.nan)
        out["entry_threshold"] = out["forecast"].shift(1).rolling(
            params.percentile_window).quantile(params.entry_quantile)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        forecasts = indicators["forecast"].to_numpy(dtype=float)
        thresholds = indicators["entry_threshold"].to_numpy(dtype=float)
        start = max(self.warmup_bars(params), ctx.warmup_bars) - 1
        pending = False
        active = False
        armed = False
        entry = stop = np.nan
        entry_bar = 0
        for i in range(len(data)):
            if pending:
                # The previous close requested entry; this bar supplies its fill.
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry = opens[i]
                    entry_bar = i
                    stop = entry - params.trail_atr * atrs[i - 1]
                    armed = False
                    active = True
            if active:
                # Existing stops are tested before raising the stop at this close.
                # A breach requests next-open liquidation, not a stop-price fill.
                breached = np.isfinite(lows[i]) and lows[i] <= stop
                expired = i - entry_bar + 1 >= params.hold_bars
                if breached or expired or not np.isfinite(closes[i]):
                    active = False
                    continue
                if closes[i] >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(atrs[i]) and atrs[i] > 0:
                        stop = max(stop, closes[i] - params.trail_atr * atrs[i])
                raw[i] = 1
            elif (i >= start and np.isfinite(forecasts[i])
                  and np.isfinite(thresholds[i]) and np.isfinite(atrs[i])
                  and atrs[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0
                  and forecasts[i] > max(0.0, thresholds[i])):
                pending = True
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GeneratedParams(pattern_window=2, shape_change=2, analog_window=3,
                             neighbors=2, percentile_window=2)
    strategy = GeneratedStrategy()
    decision = strategy.warmup_bars(params) - 1
    index = pd.date_range("2020-01-01", periods=decision + 8, freq="B")
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    bars.iloc[decision + 1, :4] = [110.0, 112.0, 109.0, 111.0]
    bars.iloc[decision + 2, :4] = [111.0, 113.0, 110.5, 112.0]
    bars.iloc[decision + 3, :4] = [112.0, 113.0, 109.5, 111.0]
    features = pd.DataFrame({"atr": 1.0, "forecast": np.nan,
                             "entry_threshold": 0.0}, index=index)
    features.iloc[decision, features.columns.get_loc("forecast")] = 0.02
    context = StrategyContext(symbol="SPY", timeframe="1d",
                              warmup_bars=strategy.warmup_bars(params))
    signals = strategy.generate_signals(bars, features, context, params).data
    # Verifies the one-bar delay, next-open entry basis, and delayed stop exit.
    assert signals["signal"].iloc[decision:decision + 5].tolist() == [0, 1, 1, 1, 0]
    assert signals["size"].gt(0).all()
