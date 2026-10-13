from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    frac_order: float = 0.45
    frac_window: int = 40
    equilibrium_window: int = 30
    regime_window: int = 40
    regime_threshold: float = 0.10
    entry_z: float = 1.20
    atr_window: int = 14
    breakeven_pct: float = 0.02
    trail_atr: float = 2.50
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790880793"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(
            params.frac_window + params.equilibrium_window + params.regime_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        p = params
        windows = (p.frac_window, p.equilibrium_window, p.regime_window, p.atr_window, p.hold_bars)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in windows):
            raise ValueError("Windows and holding period must be positive integers.")
        if min(p.frac_window, p.equilibrium_window, p.regime_window) < 3:
            raise ValueError("Fractional, equilibrium and regime windows must be at least three.")
        values = (p.frac_order, p.regime_threshold, p.entry_z, p.breakeven_pct, p.trail_atr)
        if not np.isfinite(values).all() or not (
            0 < p.frac_order < 1 and 0 < p.regime_threshold < 1
            and min(p.entry_z, p.breakeven_pct, p.trail_atr) > 0
        ):
            raise ValueError("Invalid fractional order, threshold or risk parameter.")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        weights = np.ones(p.frac_window, dtype=float)
        for k in range(1, p.frac_window):
            weights[k] = -weights[k - 1] * (p.frac_order - k + 1) / k
        fractional = np.full(len(data), np.nan)
        if len(data) >= p.frac_window:
            fractional[p.frac_window - 1:] = np.convolve(
                close.to_numpy(), weights, mode="valid"
            )
        out["fractional_close"] = fractional
        out["equilibrium"] = out["fractional_close"].rolling(p.equilibrium_window).mean()
        out["scale"] = out["fractional_close"].rolling(
            p.equilibrium_window
        ).std(ddof=0).replace(0.0, np.nan)
        out["stretch"] = (out["fractional_close"] - out["equilibrium"]) / out["scale"]
        out["velocity"] = out["fractional_close"].diff() / out["scale"].shift(1)
        out["response"] = out["stretch"].shift(1).rolling(
            p.regime_window
        ).corr(out["velocity"]).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        # Hysteresis: retain the last regime inside the neutral response band.
        regime = pd.Series(np.nan, index=data.index)
        regime.loc[out["response"] > p.regime_threshold] = 1.0
        regime.loc[out["response"] < -p.regime_threshold] = -1.0
        out["regime"] = regime.ffill().fillna(0.0)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(p.atr_window).mean()
        direction = np.sign(out["stretch"]) * out["regime"]
        ready = np.isfinite(out[["stretch", "velocity", "response", "atr"]]).all(axis=1)
        qualified = (
            ready & (out["atr"] > 0) & (out["stretch"].abs() >= p.entry_z)
            & (direction * out["velocity"] > 0)
        )
        out["candidate"] = direction.where(qualified, 0).fillna(0).astype(int)
        out["conviction"] = (
            (out["stretch"].abs() / 3.0).clip(0.0, 1.0)
            * (out["response"].abs() / 0.4).clip(0.0, 1.0)
        ).fillna(0.0)
        out["entry_size"] = 0.20 + 0.75 * out["conviction"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.20, dtype=float)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        entry_sizes = indicators["entry_size"].fillna(0.20).to_numpy(dtype=float)
        position, entry_bar = 0, -1
        entry, stop, peak, held_size = np.nan, 0.0, 0.0, 0.20
        armed = False
        warmup = self.warmup_bars(params)

        for i in range(warmup, n):
            valid = np.isfinite(prices[i]).all() and np.all(prices[i] > 0)
            if position:
                if not valid:
                    signals[i], sizes[i] = position, held_size
                    continue
                if np.isnan(entry):
                    # Yesterday's decision is filled at today's open.
                    entry, entry_bar = prices[i, 0], i
                    stop = -params.trail_atr * atr[i - 1]
                    peak = 0.0
                adverse_price = prices[i, 2] if position == 1 else prices[i, 1]
                favorable_price = prices[i, 1] if position == 1 else prices[i, 2]
                adverse = position * (adverse_price - entry)
                favorable = position * (favorable_price - entry)
                if adverse <= stop or i - entry_bar + 1 >= params.hold_bars:
                    position, entry, armed = 0, np.nan, False
                    continue
                peak = max(peak, favorable)
                if peak >= params.breakeven_pct * entry:
                    armed = True
                if armed:
                    # Signed profit coordinates: this stop only ever increases.
                    # For shorts, the corresponding price stop only decreases.
                    stop = max(stop, 0.0)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        stop = max(stop, peak - params.trail_atr * atr[i])
                signals[i], sizes[i] = position, held_size
            elif valid and candidates[i] and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(candidates[i])
                entry, armed = np.nan, False
                held_size = float(np.clip(entry_sizes[i], 0.20, 0.95))
                signals[i], sizes[i] = position, held_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    p = StrategyParams(frac_window=4, equilibrium_window=3, regime_window=3, atr_window=2)
    w = strategy.warmup_bars(p)
    index = pd.date_range("2020-01-01", periods=40)
    base = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=w)
    for direction in (1, -1):
        data = base.copy()
        data.loc[index[w + 2], "high"] = 103.0
        data.loc[index[w + 3], ["high", "low"]] = [102.0, 100.25]
        if direction == -1:
            data[["open", "close"]] = 200.0 - data[["open", "close"]]
            high, low = data["high"].copy(), data["low"].copy()
            data["high"], data["low"] = 200.0 - low, 200.0 - high
        ind = pd.DataFrame({"candidate": 0, "atr": 1.0, "entry_size": 0.80}, index=index)
        ind.loc[index[w], "candidate"] = direction
        result = strategy.generate_signals(data, ind, ctx, p).data
        expected = np.zeros(len(index), dtype=int)
        expected[w + 1:w + 4] = direction
        assert np.array_equal(result["signal"].to_numpy(), expected)
        assert result["size"].iloc[w + 1] == 0.80
        assert result["size"].gt(0).all()
        timed = strategy.generate_signals(base, ind, ctx, p).data
        assert timed["signal"].iloc[w + 1:w + p.hold_bars + 1].eq(direction).all()
        assert timed["signal"].iloc[w + p.hold_bars + 1] == 0
    rng = np.random.default_rng(17)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 240)))
    sample = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000.0,
    }, index=pd.date_range("2021-01-01", periods=len(close)))
    defaults = StrategyParams()
    full = strategy.indicators(sample, defaults)
    prefix = strategy.indicators(sample.iloc[:190], defaults)
    pd.testing.assert_frame_equal(full.iloc[:190], prefix)
    full_signals = strategy.generate_signals(sample, full, ctx, defaults).data
    prefix_signals = strategy.generate_signals(sample.iloc[:190], prefix, ctx, defaults).data
    pd.testing.assert_frame_equal(full_signals.iloc[:190], prefix_signals)
    assert set(full_signals["signal"].unique()) <= {-1, 0, 1}
    assert np.isfinite(full_signals["size"]).all()


if __name__ == "__main__":
    _self_check()
