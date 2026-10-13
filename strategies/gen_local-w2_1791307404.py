from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791307404"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        # 504 historical endpoints plus seven preceding shape bars.
        return 512

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        opening = data["open"].astype(float).to_numpy()
        price = close.to_numpy()
        log_price = np.log(close.where(close.gt(0))).replace(
            [np.inf, -np.inf], np.nan
        )
        valid = log_price.notna().rolling(8, min_periods=8).sum().eq(8)
        nodes = [log_price.shift(7 - j) for j in range(8)]
        out = pd.DataFrame(index=data.index)
        # Adjacent edges are universal; retain the 21 informative edges.
        for left in range(6):
            for right in range(left + 2, 8):
                visible = valid.copy()
                for middle in range(left + 1, right):
                    chord = nodes[left] + (nodes[right] - nodes[left]) * (
                        (middle - left) / (right - left)
                    )
                    visible &= nodes[middle].lt(chord)
                out[f"edge_{left}_{right}"] = visible.astype(float).where(valid)

        topology = out.to_numpy()
        forecast = np.full(len(data), np.nan)
        win_rate = np.full(len(data), np.nan)
        analog_count = np.zeros(len(data), dtype=int)
        # ponytail: bounded brute-force Hamming search; index signatures if
        # intraday datasets make this 504-pattern scan expensive.
        for t in range(self.warmup_bars(params), len(data)):
            if not np.isfinite(topology[t]).all():
                continue
            # j + 8 < t - 7: outcomes end before the current shape starts.
            candidates = np.arange(max(7, t - 504), t - 15)
            eligible = (
                np.isfinite(topology[candidates]).all(axis=1)
                & np.isfinite(opening[candidates + 1])
                & (opening[candidates + 1] > 0)
                & np.isfinite(price[candidates + 8])
                & (price[candidates + 8] > 0)
            )
            candidates = candidates[eligible]
            distances = np.count_nonzero(
                topology[candidates] != topology[t], axis=1
            )
            near = distances <= 4
            candidates, distances = candidates[near], distances[near]
            selected = candidates[np.argsort(distances, kind="stable")[:16]]
            analog_count[t] = len(selected)
            if len(selected) < 12:
                continue
            outcomes = price[selected + 8] / opening[selected + 1] - 1.0
            forecast[t] = np.median(outcomes)
            win_rate[t] = np.mean(outcomes > 0.0)

        out["analog_count"] = analog_count
        out["analog_median_return"] = forecast
        out["analog_win_rate"] = win_rate
        out["entry"] = (
            out["analog_median_return"].gt(0.006)
            & out["analog_win_rate"].ge(0.625)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        if (
            isinstance(params.max_hold_bars, bool)
            or int(params.max_hold_bars) != params.max_hold_bars
            or params.max_hold_bars < 1
        ):
            raise ValueError("max_hold_bars must be a positive integer")
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for t in range(len(data)):
            if active:
                if t == entry_bar:
                    # This open is read only after the preceding entry decision.
                    entry_price = opening[t]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                hit_target = (
                    valid_entry
                    and np.isfinite(closing[t])
                    and closing[t] >= entry_price * (1.0 + params.profit_target)
                )
                if (
                    not valid_entry
                    or hit_target
                    or t - entry_bar + 1 >= params.max_hold_bars
                ):
                    active = False
            elif entry[t]:
                active = True
                entry_bar = t + 1
                entry_price = np.nan
            signal[t] = int(active)

        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=14, freq="D")
    bars = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 99.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    entries = pd.DataFrame({"entry": [True] + [False] * 13}, index=index)
    timed = strategy.generate_signals(bars, entries, None, Params(0.5, 8)).data
    assert timed["signal"].tolist() == [0] + [1] * 8 + [0] * 5
    bars.loc[index[3], "close"] = 103.0
    targeted = strategy.generate_signals(bars, entries, None, Params(0.02, 8)).data
    assert targeted["signal"].tolist() == [0, 1, 1, 1] + [0] * 10
    rng = np.random.default_rng(7)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 624)))
    history = pd.DataFrame(
        {"open": prices * 0.999, "high": prices * 1.01,
         "low": prices * 0.99, "close": prices, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=624, freq="D"),
    )
    full = strategy.indicators(history, Params())
    prefix = strategy.indicators(history.iloc[:600], Params())
    pd.testing.assert_frame_equal(full.iloc[:600], prefix)
    empty = history.iloc[:0]
    assert strategy.generate_signals(
        empty, strategy.indicators(empty, Params()), None, Params()
    ).data.empty
