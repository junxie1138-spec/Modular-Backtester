from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    slope_window: int = 10
    peak_window: int = 30
    percentile_window: int = 120
    recovery_percentile: float = 0.85
    drawdown_percentile: float = 0.25
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1790970488"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.slope_window + 2, params.peak_window,
                       params.atr_window + 1) + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.slope_window, params.peak_window,
               params.percentile_window, params.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (0 < params.drawdown_percentile < 0.5
                and 0.5 < params.recovery_percentile < 1):
            raise ValueError("Percentiles must select lower drawdowns and upper innovations.")
        if params.stop_atr <= 0 or not 3 <= params.hold_bars <= 5:
            raise ValueError("Use a positive stop and a three-to-five-session hold.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_close"] = np.log(close)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        n = params.slope_window
        time = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        out["log_sum"] = out["log_close"].rolling(n).sum()
        out["time_log_sum"] = (time * out["log_close"]).rolling(n).sum()
        out["slope"] = (
            out["time_log_sum"] - (time - (n - 1) / 2.0) * out["log_sum"]
        ) / (n * (n * n - 1) / 12.0)
        out["predicted_slope"] = 2.0 * out["slope"].shift(1) - out["slope"].shift(2)
        out["relative_atr"] = (out["atr"] / close).where(out["atr"] > 0)
        out["innovation"] = (
            out["slope"] - out["predicted_slope"]
        ) / out["relative_atr"]
        out["peak"] = close.rolling(params.peak_window).max()
        out["drawdown"] = np.log(close / out["peak"])
        out["innovation_threshold"] = out["innovation"].rolling(
            params.percentile_window
        ).quantile(params.recovery_percentile).shift(1)
        out["drawdown_threshold"] = out["drawdown"].rolling(
            params.percentile_window
        ).quantile(params.drawdown_percentile).shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["drawdown"] < out["drawdown_threshold"])
            & (out["slope"] < 0)
            & (out["innovation"] > out["innovation_threshold"].clip(lower=0))
            & (out["innovation"].shift(1)
               <= out["innovation_threshold"].shift(1).clip(lower=0))
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RecoveryParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        frozen_atr = stop = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Freeze the stop using the next-open entry and decision-bar ATR.
                    stop = opens[i] - params.stop_atr * frozen_atr
                if (not np.isfinite(stop) or not np.isfinite(closes[i])
                        or closes[i] < stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    active = False
                else:
                    raw[i] = 1
            elif (entries[i] and np.isfinite(atr[i]) and atr[i] > 0
                  and np.isfinite(closes[i]) and closes[i] > 0):
                active = True
                entry_bar = i + 1
                frozen_atr = atr[i]
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 105.0, "low": 96.0,
         "close": [100.0, 104.0, 101.0, 97.0, 100.0, 100.0, 100.0],
         "volume": 1000.0}, index=index
    )
    indicators = pd.DataFrame(
        {"entry": [True, False, False, False, False, False, False],
         "atr": 1.0}, index=index
    )
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d",
                          warmup_bars=strategy.warmup_bars(params))
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    data["close"] = 100.0
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert (signals["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
