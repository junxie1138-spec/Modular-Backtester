from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    trend_span: int = 12
    innovation_threshold: float = 0.6
    vol_window: int = 20
    atr_window: int = 14
    target_vol: float = 0.12
    breakeven_pct: float = 0.004
    trail_atr: float = 1.2
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791280487"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200, 2 * params.trend_span + 1,
                       params.vol_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if not (params.trend_span >= 2 and params.vol_window >= 2
                and params.atr_window >= 1 and 1 <= params.max_hold <= 2
                and params.innovation_threshold > 0 and params.target_vol > 0
                and params.breakeven_pct > 0 and params.trail_atr > 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["smooth_fast"] = out["log_close"].ewm(
            span=params.trend_span, adjust=False,
            min_periods=params.trend_span).mean()
        out["smooth_slow"] = out["smooth_fast"].ewm(
            span=params.trend_span, adjust=False,
            min_periods=params.trend_span).mean()
        alpha = 2.0 / (params.trend_span + 1.0)
        out["latent_level"] = 2.0 * out["smooth_fast"] - out["smooth_slow"]
        out["latent_slope"] = alpha / (1.0 - alpha) * (
            out["smooth_fast"] - out["smooth_slow"])
        out["forecast"] = (
            out["latent_level"] + out["latent_slope"]).shift(1)
        out["innovation"] = out["log_close"] - out["forecast"]
        out["log_return"] = out["log_close"].diff()
        out["daily_vol"] = out["log_return"].rolling(
            params.vol_window, min_periods=params.vol_window).std(ddof=0)
        out["innovation_scale"] = out["daily_vol"].shift(1).replace(0, np.nan)
        out["innovation_score"] = out["innovation"] / out["innovation_scale"]
        out["annual_vol"] = out["daily_vol"] * np.sqrt(252.0)
        out["exposure"] = (params.target_vol / out["annual_vol"].replace(
            0, np.nan)).clip(0.05, 1.0)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        ready = out[["innovation_score", "latent_slope", "ma200",
                     "atr", "exposure"]].notna().all(axis=1)
        out["candidate"] = np.select([
            ready & (close > out["ma200"]) & (out["latent_slope"] > 0)
            & (out["innovation_score"] > params.innovation_threshold),
            ready & (close < out["ma200"]) & (out["latent_slope"] < 0)
            & (out["innovation_score"] < -params.innovation_threshold),
        ], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        ohlc = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        values = indicators[["candidate", "atr", "ma200", "exposure"]].to_numpy(
            dtype=float)
        side = pending = age = 0
        entry = best = stop = 0.0
        held_size = 1.0
        armed = False
        for i in range(len(data)):
            opening, high, low, close = ohlc[i]
            candidate, atr, ma200, exposure = values[i]
            valid = (np.isfinite(ohlc[i]).all()
                     and np.isfinite(values[i]).all()
                     and min(opening, high, low, close) > 0 and atr > 0)
            if not valid:
                side = pending = 0
                continue
            if pending:
                # A decision on the preceding close enters at this open.
                side, pending, age = pending, 0, 0
                entry = opening
                best = side * entry
                stop = best - params.trail_atr * values[i - 1, 1]
                armed = False
            if side:
                age += 1
                # Signed prices make both long and short stops ratchet upward.
                adverse = low if side == 1 else -high
                hit = adverse <= stop
                favorable = high if side == 1 else -low
                best = max(best, favorable)
                if best - side * entry >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, side * entry, best - params.trail_atr * atr)
                    # New stops are checked against the close, avoiding guesses
                    # about the ordering of today's high and low.
                    hit = hit or side * close <= stop
                regime_exit = side * (close - ma200) <= 0
                if hit or regime_exit or age >= params.max_hold:
                    side = 0
                signals[i] = side
                sizes[i] = held_size
            elif candidate:
                pending = int(candidate)
                signals[i] = pending
                held_size = float(exposure)
                sizes[i] = held_size
        df["signal"] = signals
        df["size"] = pd.Series(sizes, index=data.index).shift(1).fillna(1.0).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=4)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    for direction in (1, -1):
        data = pd.DataFrame({
            "open": [100.0] * 4,
            "high": [100.2, 101.0 if direction == 1 else 100.2, 100.2, 100.2],
            "low": [99.8, 99.8 if direction == 1 else 99.0, 99.8, 99.8],
            "close": [100.0, 99.9 if direction == 1 else 100.1, 100.0, 100.0],
            "volume": [1000.0] * 4,
        }, index=index)
        ind = pd.DataFrame({
            "candidate": [direction, 0, 0, 0], "atr": [2.0] * 4,
            "ma200": [90.0 if direction == 1 else 110.0] * 4,
            "exposure": [0.4] * 4,
        }, index=index)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, direction, 0, 0]
        assert result["size"].gt(0).all()
        # A quiet entry survives one session and expires after the second.
        data["high"], data["low"], data["close"] = 100.2, 99.8, 100.0
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, 0]
        prefix = strategy.generate_signals(data.iloc[:3], ind.iloc[:3], ctx, params).data
        pd.testing.assert_frame_equal(result.iloc[:3], prefix)


if __name__ == "__main__":
    _self_check()
