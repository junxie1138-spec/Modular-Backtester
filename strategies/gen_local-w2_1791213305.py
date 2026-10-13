from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    shape_window: int = 20
    archive_bars: int = 504
    analog_count: int = 12
    max_distance: float = 0.22
    min_forecast: float = 0.012
    min_win_rate: float = 0.60
    profit_target: float = 0.04
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791213305"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        if not (
            params.shape_window >= 3
            and params.archive_bars >= params.analog_count >= 1
            and params.hold_bars >= 1
            and params.profit_target > 0
            and params.max_distance > 0
            and 0 <= params.min_win_rate <= 1
        ):
            raise ValueError("Invalid strategy parameters")
        return int(params.archive_bars + 2 * params.shape_window + params.hold_bars + 3)

    def indicators(self, data, params):
        warmup = self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        a, b, c = close.shift(2), close.shift(1), close
        valid = a.notna() & b.notna() & c.notna()
        # Chronological order breaks equal-price ties consistently.
        out["pattern"] = (
            4 * (a <= b).astype(int)
            + 2 * (a <= c).astype(int)
            + (b <= c).astype(int)
        ).where(valid)
        cols = []
        for code in (0, 1, 3, 4, 6, 7):
            col = f"frequency_{code}"
            out[col] = (
                out["pattern"].eq(code).astype(float).where(valid)
                .rolling(params.shape_window, min_periods=params.shape_window).mean()
            )
            cols.append(col)
        frequencies = out[cols]
        out["complexity"] = -(
            frequencies * np.log(frequencies.clip(lower=1e-12))
        ).sum(axis=1, min_count=6) / np.log(6.0)
        entry_open = data["open"].where(data["open"] > 0).shift(params.hold_bars - 1)
        out["matured_return"] = close / entry_open - 1.0
        roots = np.sqrt(frequencies.to_numpy(dtype=float))
        entropy = out["complexity"].to_numpy(dtype=float)
        outcomes = out["matured_return"].to_numpy(dtype=float)
        forecasts = np.full(len(data), np.nan)
        wins = np.full(len(data), np.nan)
        distances = np.full(len(data), np.nan)
        # ponytail: O(bars * archive_bars); shorten the archive for very large histories.
        for i in range(warmup, len(data)):
            if not np.isfinite(entropy[i]):
                continue
            # Analog outcomes finish before the current shape window begins.
            end = i - params.hold_bars - params.shape_window - 2
            candidates = np.arange(max(0, end - params.archive_bars + 1), end + 1)
            labels = outcomes[candidates + params.hold_bars]
            distance = (
                np.sqrt(((roots[candidates] - roots[i]) ** 2).sum(axis=1) / 2.0)
                + np.abs(entropy[candidates] - entropy[i])
            )
            usable = (
                np.isfinite(distance) & np.isfinite(labels)
                & (distance <= params.max_distance)
            )
            if int(usable.sum()) < params.analog_count:
                continue
            nearest = np.argsort(distance[usable])[:params.analog_count]
            selected = labels[usable][nearest]
            forecasts[i] = selected.mean()
            wins[i] = (selected > 0).mean()
            distances[i] = distance[usable][nearest].mean()
        out["forecast"] = forecasts
        out["win_rate"] = wins
        out["analog_distance"] = distances
        out["entry_candidate"] = (
            out["forecast"].ge(params.min_forecast)
            & out["win_rate"].ge(params.min_win_rate)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        warmup = self.warmup_bars(params)
        candidate = indicators["entry_candidate"].fillna(False).astype(bool)
        confirmed = (candidate & candidate.shift(1, fill_value=False)).to_numpy()
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    entry_price = opens[i]
                invalid = (
                    not np.isfinite(entry_price) or entry_price <= 0
                    or not np.isfinite(closes[i])
                )
                target = (
                    not invalid
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                expired = i - entry_bar + 1 >= params.hold_bars
                if invalid or target or expired:
                    active = False
                else:
                    raw[i] = 1
            elif (
                i >= warmup and confirmed[i]
                and np.isfinite(closes[i]) and closes[i] > 0
            ):
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params(shape_window=6, archive_bars=30, analog_count=6)
    index = pd.date_range("2020-01-01", periods=220, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(
        symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params)
    )
    forced = pd.DataFrame({"entry_candidate": True}, index=index)
    first = strategy.warmup_bars(params) + 1
    signals = strategy.generate_signals(data, forced, ctx, params).data["signal"]
    assert signals.iloc[:first].eq(0).all()
    assert signals.iloc[first:first + params.hold_bars].eq(1).all()
    assert signals.iloc[first + params.hold_bars] == 0
    data.iloc[first + 2, data.columns.get_loc("close")] = 105.0
    signals = strategy.generate_signals(data, forced, ctx, params).data["signal"]
    assert signals.iloc[first + 2] == 1 and signals.iloc[first + 3] == 0
    forced["entry_candidate"] = False
    forced.iloc[first + 5, 0] = True
    assert strategy.generate_signals(data, forced, ctx, params).data["signal"].eq(0).all()
    rng = np.random.default_rng(21)
    data["close"] = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(data))))
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:180], params)
    pd.testing.assert_frame_equal(full.iloc[:180], prefix)
    assert full["complexity"].dropna().between(-1e-10, 1.0 + 1e-10).all()


if __name__ == "__main__":
    _self_check()
