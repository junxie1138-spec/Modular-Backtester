from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    trend_window: int = 8
    swing_window: int = 20
    noise_window: int = 20
    atr_window: int = 14
    depth_atr: float = 1.5
    innovation_threshold: float = 0.12
    breakeven_pct: float = 0.01
    trail_atr: float = 2.0
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1790920514"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(200, params.trend_window + 1, params.swing_window,
                       params.noise_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.trend_window, params.swing_window, params.noise_window,
               params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two.")
        if (params.depth_atr <= 0 or params.innovation_threshold <= 0
                or params.breakeven_pct <= 0 or params.trail_atr <= 0
                or params.hold_bars < 1):
            raise ValueError("Thresholds and holding period must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["log_close"] = np.log(close.where(close > 0))
        out["log_return"] = out["log_close"].diff()
        out["noise"] = out["log_return"].rolling(params.noise_window).std()
        n = params.trend_window
        x = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        center = x - (n - 1) / 2.0
        # Rolling OLS slope estimates the latent drift of log price.
        out["slope"] = (
            (x * out["log_close"]).rolling(n).sum()
            - center * out["log_close"].rolling(n).sum()
        ) / (n * (n * n - 1) / 12.0)
        out["innovation"] = out["slope"].diff() / out["noise"].replace(0, np.nan)
        out["peak"] = close.rolling(params.swing_window).max()
        out["trough"] = close.rolling(params.swing_window).min()
        atr = out["atr"].where(out["atr"] > 0)
        out["drawdown_atr"] = (out["peak"] - close) / atr
        out["runup_atr"] = (close - out["trough"]) / atr
        innovation = out["innovation"]
        threshold = params.innovation_threshold
        out["long_entry"] = (
            (close > out["ma200"]) & (out["drawdown_atr"] >= params.depth_atr)
            & (out["slope"] < 0) & (innovation > threshold)
            & (innovation.shift(1) <= threshold)
        ).fillna(False)
        out["short_entry"] = (
            (close < out["ma200"]) & (out["runup_atr"] >= params.depth_atr)
            & (out["slope"] > 0) & (innovation < -threshold)
            & (innovation.shift(1) >= -threshold)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RecoveryParams
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        side = 0
        pending = 0
        age = 0
        entry = best = stop = 0.0
        armed = False
        for i in range(len(data)):
            exited = False
            if pending:
                # The previous close's order fills at this bar's open.
                if np.isfinite(opens[i]) and opens[i] > 0 and i > 0:
                    side = pending
                    entry = opens[i]
                    best = side * entry
                    stop = best - params.trail_atr * atrs[i - 1]
                    age = 0
                    armed = False
                pending = 0
            if side:
                age += 1
                if not (np.isfinite(closes[i]) and closes[i] > 0
                        and np.isfinite(atrs[i]) and atrs[i] > 0):
                    side = 0
                    exited = True
                else:
                    signed_close = side * closes[i]
                    best = max(best, signed_close)
                    if best - side * entry >= params.breakeven_pct * entry:
                        armed = True
                    if armed:
                        # Signed coordinates ratchet upward for both directions.
                        stop = max(stop, side * entry, best - params.trail_atr * atrs[i])
                    if signed_close <= stop or age >= params.hold_bars:
                        side = 0
                        exited = True
            if not side and not exited and np.isfinite(atrs[i]) and atrs[i] > 0:
                pending = 1 if longs[i] else (-1 if shorts[i] else 0)
            signals[i] = side if side else pending
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    index = pd.date_range("2020-01-01", periods=7)
    for side in (1, -1):
        for trail in (True, False):
            close = np.array([100., 100., 102., 100.5, 100., 100., 100.])
            if not trail:
                close[:] = 100.
            close = 100. + side * (close - 100.)
            data = pd.DataFrame({"open": 100., "close": close}, index=index)
            ind = pd.DataFrame({
                "atr": 0.5,
                "long_entry": [side == 1] + [False] * 6,
                "short_entry": [side == -1] + [False] * 6,
            }, index=index)
            result = strategy.generate_signals(data, ind, None, params).data
            expected = ([0, side, side, side, 0, 0, 0] if trail
                        else [0, side, side, side, side, 0, 0])
            assert result["signal"].tolist() == expected
            assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
