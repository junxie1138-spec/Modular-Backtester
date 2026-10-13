from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    streak_days: int = 3
    energy_window: int = 15
    energy_ratio: float = 1.4
    vol_window: int = 20
    atr_window: int = 14
    target_vol: float = 0.12
    breakeven_pct: float = 0.025
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791341925"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.streak_days, params.energy_window,
                       params.vol_window, params.atr_window) + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.streak_days, params.energy_window, params.vol_window,
               params.atr_window, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if min(params.target_vol, params.energy_ratio,
               params.breakeven_pct, params.trail_atr) <= 0:
            raise ValueError("Risk and entry thresholds must be positive")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["higher_low"] = data["low"].gt(data["low"].shift(1))
        runs = (~out["higher_low"]).cumsum()
        out["streak"] = out["higher_low"].astype(int).groupby(runs).cumsum().clip(upper=params.streak_days)
        out["up_energy"] = out["return"].clip(lower=0).pow(2).rolling(params.energy_window).mean()
        out["down_energy"] = out["return"].clip(upper=0).pow(2).rolling(params.energy_window).mean()
        out["dominance"] = out["up_energy"] / (out["down_energy"] + 1e-12)
        out["annual_vol"] = out["return"].rolling(params.vol_window).std(ddof=0) * np.sqrt(252.0)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["ready"] = (
            out[["up_energy", "down_energy", "annual_vol", "atr"]].notna().all(axis=1)
            & out["annual_vol"].gt(0) & out["atr"].gt(0)
        )
        out["size"] = (params.target_vol / out["annual_vol"].where(out["annual_vol"] > 0)).clip(0.05, 1.0).fillna(1.0)
        out["entry"] = (
            out["ready"] & out["streak"].ge(params.streak_days)
            & out["dominance"].ge(params.energy_ratio)
        )
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=bool)
        weights = indicators["size"].to_numpy(dtype=float)
        pending = False
        active = False
        armed = False
        entry = stop = peak = 0.0
        weight = 1.0
        age = 0
        for i in range(n):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry = peak = opens[i]
                    stop = entry - params.trail_atr * atr[i - 1]
                    active = True
                    armed = False
                    age = 0
            if active:
                age += 1
                size[i] = weight
                # Test the previously established stop; new ratchets apply next bar.
                if lows[i] <= stop or age >= params.hold_bars:
                    active = False
                    continue
                peak = max(peak, highs[i])
                if peak >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                signal[i] = 1
            elif entries[i]:
                weight = float(np.clip(weights[i], 0.05, 1.0))
                size[i] = weight
                signal[i] = 1
                pending = True
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100, 100, 103, 100, 100],
        "high": [101, 104, 105, 101, 101],
        "low": [99, 99, 103, 99, 99],
        "close": [100, 103, 104, 100, 100],
        "volume": [1000] * 5,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=21)
    indicators = pd.DataFrame({
        "entry": [True, False, False, False, False],
        "atr": [1.0] * 5, "size": [0.5] * 5,
    }, index=index)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    actual = strategy.indicators(data, params)
    assert not actual["entry"].any()
    assert np.isfinite(actual.select_dtypes(include="number").to_numpy()).all()
    assert strategy.warmup_bars(params) >= 21
    timed = strategy.generate_signals(data, indicators, ctx, Params(hold_bars=2)).data
    assert timed["signal"].tolist() == [0, 1, 1, 0, 0]
    for end in range(1, len(data) + 1):
        prefix = strategy.generate_signals(data.iloc[:end], indicators.iloc[:end], ctx, params).data
        pd.testing.assert_frame_equal(prefix, result.iloc[:end])
