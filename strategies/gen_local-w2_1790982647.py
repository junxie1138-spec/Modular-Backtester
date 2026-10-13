from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnRegimeParams:
    regime_window: int = 60
    aggregation: int = 5
    impulse_window: int = 3
    volatility_window: int = 20
    atr_window: int = 14
    regime_gap: float = 0.20
    impulse_threshold: float = 0.70
    spike_multiple: float = 3.0
    refractory_bars: int = 3
    breakeven_pct: float = 0.01
    initial_atr: float = 2.5
    trail_atr: float = 2.0
    max_hold: int = 10
    entry_slippage_bps: float = 5.0


class GeneratedStrategy(BaseStrategy[ReturnRegimeParams]):
    strategy_id = "gen_local-w2_1790982647"

    @classmethod
    def params_type(cls):
        return ReturnRegimeParams

    def warmup_bars(self, params):
        return int(max(
            params.regime_window + params.aggregation,
            params.volatility_window + params.refractory_bars,
            params.impulse_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data, params):
        windows = (
            params.regime_window, params.aggregation, params.impulse_window,
            params.volatility_window, params.atr_window,
            params.refractory_bars, params.max_hold,
        )
        if any(w < 2 for w in windows):
            raise ValueError("All windows and holding limits must be at least two.")
        if not (0 < params.regime_gap < 1):
            raise ValueError("regime_gap must lie between zero and one.")
        if any(not np.isfinite(x) or x <= 0 for x in (
            params.impulse_threshold, params.spike_multiple,
            params.breakeven_pct, params.initial_atr, params.trail_atr,
        )):
            raise ValueError("Signal and stop thresholds must be finite and positive.")
        if not (0 <= params.entry_slippage_bps < 10000):
            raise ValueError("Invalid entry slippage.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["aggregate_return"] = out["return"].rolling(params.aggregation).sum()
        out["daily_variance"] = out["return"].rolling(params.regime_window).var(ddof=0)
        out["aggregate_variance"] = out["aggregate_return"].rolling(params.regime_window).var(ddof=0)
        out["variance_ratio"] = out["aggregate_variance"] / (
            params.aggregation * out["daily_variance"].where(out["daily_variance"] > 0)
        )
        out["volatility"] = out["return"].rolling(params.volatility_window).std(ddof=0)
        out["impulse_return"] = out["return"].rolling(params.impulse_window).sum()
        out["impulse"] = out["impulse_return"] / (
            np.sqrt(params.impulse_window) * out["volatility"].where(out["volatility"] > 0)
        )
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["spike"] = (
            out["return"].abs() > params.spike_multiple * out["volatility"].shift(1)
        ).astype(float)
        out["recent_spike"] = out["spike"].rolling(params.refractory_bars).max()
        out["regime"] = np.select([
            out["variance_ratio"] > 1 + params.regime_gap,
            out["variance_ratio"] < 1 - params.regime_gap,
        ], [1, -1], default=0).astype(int)
        valid = (
            np.isfinite(out["variance_ratio"]) & np.isfinite(out["impulse"])
            & np.isfinite(out["atr"]) & (out["atr"] > 0)
            & out["recent_spike"].eq(0)
            & out["impulse"].abs().ge(params.impulse_threshold)
        )
        out["candidate"] = np.where(
            valid, np.sign(out["impulse"].fillna(0)) * out["regime"], 0
        ).astype(int)
        out["confirmed"] = np.where(
            out["candidate"].eq(out["candidate"].shift(1)),
            out["candidate"], 0,
        ).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(0).to_numpy(dtype=int)
        side = 0
        pending = False
        armed = False
        entry = stop = entry_atr = 0.0
        age = 0
        next_entry_bar = 0
        for i in range(len(data)):
            if side:
                if pending:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        pending = False
                        next_entry_bar = i + 2
                        continue
                    fill = opens[i] * (1 + side * params.entry_slippage_bps / 10000)
                    entry = side * fill
                    stop = entry - params.initial_atr * entry_atr
                    pending = False
                age += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0 or not np.isfinite(atr[i]):
                    side = 0
                    next_entry_bar = i + 2
                    continue
                # Signed prices let both long and short stops ratchet upward.
                price = side * closes[i]
                if price <= stop or age >= params.max_hold:
                    side = 0
                    next_entry_bar = i + 2
                else:
                    if price - entry >= abs(entry) * params.breakeven_pct:
                        armed = True
                    if armed:
                        stop = max(stop, entry, price - params.trail_atr * atr[i])
            elif i >= next_entry_bar and confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0:
                side = int(confirmed[i])
                pending = True
                armed = False
                age = 0
                entry_atr = atr[i]
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    rng = np.random.default_rng(17)
    index = pd.date_range("2020-01-01", periods=600, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, len(index))))
    opening = np.r_[close[0], close[:-1]]
    data = pd.DataFrame({
        "open": opening, "high": np.maximum(opening, close) * 1.003,
        "low": np.minimum(opening, close) * 0.997,
        "close": close, "volume": 1000000.0,
    }, index=index)
    strategy = GeneratedStrategy()
    params = ReturnRegimeParams()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    features = strategy.indicators(data, params)
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result.index.equals(data.index)
    assert result["signal"].isin([-1, 0, 1]).all()
    assert result["size"].gt(0).all()
    assert result["signal"].iloc[:ctx.warmup_bars].eq(0).all()
    assert result["signal"].ne(0).any()
    prefix = data.iloc[:350]
    prefix_result = strategy.generate_signals(
        prefix, strategy.indicators(prefix, params), ctx, params
    ).data
    pd.testing.assert_frame_equal(result.iloc[:350], prefix_result)
    entries = result["signal"].ne(0) & result["signal"].shift(1).fillna(0).eq(0)
    expected = features["confirmed"].shift(1).fillna(0).astype(int)
    assert result.loc[entries, "signal"].eq(expected.loc[entries]).all()
