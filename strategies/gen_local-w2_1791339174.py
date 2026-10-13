from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    shape_bars: int = 8
    history_bars: int = 252
    atr_bars: int = 14
    neighbors: int = 12
    max_mismatch: float = 0.28
    entry_threshold: float = 0.18
    stop_atr: float = 1.2
    sizing_scale: float = 1.0


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local-w2_1791339174"

    @classmethod
    def params_type(cls):
        return TopologyParams

    def warmup_bars(self, params: TopologyParams) -> int:
        return int(max(params.history_bars + params.shape_bars + 2, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        p = params
        if (p.shape_bars < 3 or p.history_bars < p.neighbors or p.neighbors < 6
                or p.atr_bars < 2 or not 0 < p.max_mismatch <= 1
                or p.entry_threshold <= 0 or p.stop_atr <= 0 or p.sizing_scale <= 0):
            raise ValueError("Invalid topology parameters")
        n = len(data)
        out = pd.DataFrame(index=data.index)
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        previous = data["close"].shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = tr.rolling(p.atr_bars, min_periods=p.atr_bars).mean()
        atr = out["atr"].to_numpy()
        width = (p.shape_bars - 1) * (p.shape_bars - 2)
        topology = np.full((n, width), np.nan)
        if n >= p.shape_bars:
            log_close = np.log(np.where(close > 0, close, np.nan))
            windows = np.lib.stride_tricks.sliding_window_view(log_close, p.shape_bars)
            valid = np.isfinite(windows).all(axis=1)
            column = 0
            # Nonadjacent edges retain temporal identity; both price polarities are used.
            for polarity in (1.0, -1.0):
                paths = polarity * windows
                for left in range(p.shape_bars - 2):
                    for right in range(left + 2, p.shape_bars):
                        fraction = np.arange(1, right - left) / (right - left)
                        line = paths[:, left, None] + (
                            paths[:, right, None] - paths[:, left, None]
                        ) * fraction
                        edges = (paths[:, left + 1:right] < line).all(axis=1)
                        topology[p.shape_bars - 1:, column] = np.where(valid, edges, np.nan)
                        column += 1
        features = pd.DataFrame(topology, index=data.index,
                                columns=[f"edge_{j}" for j in range(width)])
        forecast = np.full(n, np.nan)
        agreement = np.full(n, np.nan)
        mismatch = np.full(n, np.nan)
        # ponytail: bounded brute-force analog search; index fingerprints if history becomes large.
        for i in range(self.warmup_bars(p) - 1, n):
            if not np.isfinite(topology[i]).all():
                continue
            # j+2 <= i: every analog's two-session outcome is already observable.
            candidates = np.arange(max(p.shape_bars - 1, i - p.history_bars - 1), i - 1)
            good = (np.isfinite(topology[candidates]).all(axis=1)
                    & np.isfinite(atr[candidates]) & (atr[candidates] > 0)
                    & np.isfinite(opens[candidates + 1]) & (opens[candidates + 1] > 0)
                    & np.isfinite(close[candidates + 2]) & (close[candidates + 2] > 0))
            candidates = candidates[good]
            if len(candidates) < 6:
                continue
            distance = np.mean(topology[candidates] != topology[i], axis=1)
            selected = np.argsort(distance, kind="stable")[:p.neighbors]
            selected = selected[distance[selected] <= p.max_mismatch]
            if len(selected) < 6:
                continue
            analogs = candidates[selected]
            outcomes = (close[analogs + 2] - opens[analogs + 1]) / atr[analogs]
            weights = 1.0 / (1.0 / width + distance[selected])
            weights /= weights.sum()
            forecast[i] = np.dot(weights, np.clip(outcomes, -3.0, 3.0))
            agreement[i] = abs(np.dot(weights, np.sign(outcomes)))
            mismatch[i] = np.dot(weights, distance[selected])
        out = pd.concat([out, features], axis=1)
        out["forecast"] = forecast
        out["agreement"] = agreement
        out["mismatch"] = mismatch
        out["score"] = (out["forecast"] * out["agreement"]).fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: TopologyParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        score = indicators["score"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        direction = 0
        entry_bar = -1
        entry_price = np.nan
        entry_atr = np.nan
        exposure = 1.0
        for i in range(n):
            if direction:
                if i == entry_bar:
                    entry_price = opens[i]
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(close[i])
                # Fixed entry-open +/- entry-decision ATR; the level never ratchets.
                stopped = valid and direction * (close[i] - entry_price) < -params.stop_atr * entry_atr
                expired = i >= entry_bar + 1
                if not valid or stopped or expired:
                    direction = 0
                # No same-close reentry after an exit.
            elif (i >= self.warmup_bars(params) - 1
                  and np.isfinite(score[i]) and abs(score[i]) >= params.entry_threshold
                  and np.isfinite(atr[i]) and atr[i] > 0
                  and np.isfinite(close[i]) and close[i] > 0):
                direction = 1 if score[i] > 0 else -1
                entry_bar = i + 1
                entry_price = np.nan
                entry_atr = atr[i]
                exposure = float(np.clip(abs(score[i]) / params.sizing_scale, 0.1, 1.0))
            signals[i] = direction
            sizes[i] = exposure if direction else 1.0
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    rng = np.random.default_rng(17)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.008, 120)))
    data = pd.DataFrame({
        "open": np.r_[close[0], close[:-1]],
        "high": close * 1.025, "low": close * 0.975,
        "close": close, "volume": 1000000.0,
    }, index=pd.date_range("2020-01-01", periods=len(close)))
    p = TopologyParams(history_bars=40, neighbors=6)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(p))
    full = strategy.indicators(data, p)
    prefix = strategy.indicators(data.iloc[:90], p)
    pd.testing.assert_frame_equal(full.iloc[:90], prefix)
    result = strategy.generate_signals(data, full, ctx, p).data
    pd.testing.assert_frame_equal(
        result.iloc[:90], strategy.generate_signals(data.iloc[:90], prefix, ctx, p).data)
    assert result["signal"].isin([-1, 0, 1]).all()
    assert (result["size"] > 0).all()
    forced = full.copy()
    forced["score"] = 0.0
    start = strategy.warmup_bars(p) - 1
    forced.iloc[start, forced.columns.get_loc("score")] = 0.5
    held = strategy.generate_signals(data, forced, ctx, p).data
    assert held["signal"].iloc[start] == 0
    assert held["signal"].iloc[start + 1:start + 3].tolist() == [1, 1]
    assert held["signal"].iloc[start + 3] == 0
    for direction in (1, -1):
        forced.iloc[start, forced.columns.get_loc("score")] = direction * 0.5
        shocked = data.copy()
        shocked.iloc[start + 1, shocked.columns.get_loc("close")] = (
            data["open"].iloc[start + 1] - direction * 2.0 * p.stop_atr * full["atr"].iloc[start])
        stopped = strategy.generate_signals(shocked, forced, ctx, p).data
        assert stopped["signal"].iloc[start + 1] == direction
        assert stopped["signal"].iloc[start + 2] == 0
        assert stopped["size"].iloc[start + 1] == 0.5


if __name__ == "__main__":
    _self_check()
