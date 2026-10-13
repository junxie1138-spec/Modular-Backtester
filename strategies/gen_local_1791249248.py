from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    gap_threshold: float = 0.5
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791249248"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return 22

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (np.isfinite(params.gap_threshold) and params.gap_threshold > 0
                and np.isfinite(params.trail_atr) and params.trail_atr > 0):
            raise ValueError("Parameters must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0)
        close = data["close"].where(data["close"] > 0)
        previous = close.shift(1)
        out["body"] = np.log(close / opening)
        out["gap"] = np.log(opening / previous)
        out["expected_gap"] = out["gap"].rolling(20, min_periods=20).mean().shift(1)
        out["latent_slope"] = out["body"].rolling(8, min_periods=8).mean()
        out["previous_slope"] = out["latent_slope"].shift(1)
        out["slope_innovation"] = out["latent_slope"].diff()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["gap_shock"] = (out["gap"] - out["expected_gap"]) / (
            out["atr"] / previous
        ).replace(0, np.nan)
        out["recovery"] = close / previous - 1.0
        out = out.replace([np.inf, -np.inf], np.nan)
        out["ready"] = out.notna().all(axis=1) & (out["atr"] > 0)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        ready = indicators["ready"].fillna(False).astype(bool)
        long_entry = (
            ready
            & (indicators["gap_shock"] <= -params.gap_threshold)
            & (indicators["previous_slope"] > 0)
            & (indicators["latent_slope"] > 0)
            & (indicators["slope_innovation"] > 0)
            & (indicators["recovery"] > 0)
        )
        short_entry = (
            ready
            & (indicators["gap_shock"] >= params.gap_threshold)
            & (indicators["previous_slope"] < 0)
            & (indicators["latent_slope"] < 0)
            & (indicators["slope_innovation"] < 0)
            & (indicators["recovery"] < 0)
        )
        candidates = np.where(long_entry, 1, np.where(short_entry, -1, 0))
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        valid = ready.to_numpy()
        signals = np.zeros(len(data), dtype=int)
        direction = 0
        entry = None
        stop = peak = 0.0
        armed = False
        age = 0
        for i in range(len(data)):
            if direction:
                if (not valid[i] or not np.isfinite(opening[i])
                        or opening[i] <= 0 or not np.isfinite(closing[i])
                        or closing[i] <= 0):
                    direction, entry = 0, None
                    continue
                if entry is None:
                    # A decision at the previous close fills at this open.
                    entry = opening[i]
                    peak = direction * entry
                    stop = peak - params.trail_atr * atr[i - 1]
                    armed = False
                    age = 0
                age += 1
                signed_close = direction * closing[i]
                peak = max(peak, signed_close)
                armed = armed or direction * (closing[i] / entry - 1.0) >= 0.01
                if armed:
                    # Ratchet upward in profit coordinates, including shorts.
                    stop = max(stop, direction * entry, peak - params.trail_atr * atr[i])
                # Stops are evaluated at closes; the exit fills next bar.
                if signed_close <= stop or age >= 10:
                    direction, entry = 0, None
            elif candidates[i]:
                direction = int(candidates[i])
                entry = None
            signals[i] = direction
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    for direction in (1, -1):
        prices = 100.0 + direction * np.array([0.0, 0.4, 1.2, 3.0, 2.0, 0.8, 0.5, 0.5])
        index = pd.date_range("2020-01-01", periods=len(prices))
        data = pd.DataFrame({
            "open": 100.0, "high": prices + 1.0, "low": prices - 1.0,
            "close": prices, "volume": 1000.0,
        }, index=index)
        indicators = pd.DataFrame({
            "ready": True, "atr": 1.0, "gap_shock": 0.0,
            "previous_slope": direction * 0.01,
            "latent_slope": direction * 0.01,
            "slope_innovation": direction * 0.001,
            "recovery": direction * 0.01,
        }, index=index)
        indicators.loc[index[0], "gap_shock"] = -direction
        indicators.loc[index[4], "atr"] = 10.0
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, direction, 0, 0]
        assert (result["size"] > 0).all()
        prefix = strategy.generate_signals(data.iloc[:5], indicators.iloc[:5], None, params).data
        assert prefix.equals(result.iloc[:5])
    flat = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=pd.date_range("2020-01-01", periods=40))
    calculated = strategy.indicators(flat, params)
    assert not calculated["ready"].iloc[:21].any()
    assert calculated["ready"].iloc[21:].all()
    assert not strategy.generate_signals(flat, calculated, None, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
