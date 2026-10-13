from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    slope_window: int = 12
    atr_window: int = 14
    innovation_threshold: float = 0.25
    stop_atr: float = 1.25
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791247042"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.slope_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.slope_window < 3 or params.atr_window < 1:
            raise ValueError("Lookbacks must be positive; slope_window must be at least 3.")
        if not np.isfinite(params.innovation_threshold) or params.innovation_threshold <= 0:
            raise ValueError("innovation_threshold must be positive and finite.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0 or params.hold_bars not in (1, 2):
            raise ValueError("stop_atr must be positive and finite; hold_bars must be 1 or 2.")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1).where(lambda s: s > 0)
        opening = data["open"].where(lambda s: s > 0)
        closing = data["close"].where(lambda s: s > 0)
        out["gap"] = np.log(opening / previous).replace([np.inf, -np.inf], np.nan)
        out["intraday"] = np.log(closing / opening).replace([np.inf, -np.inf], np.nan)
        out["overnight_path"] = out["gap"].fillna(0.0).cumsum()
        window = params.slope_window
        weights = np.arange(window, dtype=float) - (window - 1) / 2.0
        weights /= np.dot(weights, weights)
        slope = np.full(len(data), np.nan)
        if len(data) >= window:
            slope[window - 1:] = np.convolve(out["overnight_path"].to_numpy(), weights[::-1], mode="valid")
        out["slope"] = pd.Series(slope, index=data.index).where(out["gap"].rolling(window).count() == window)
        out["gap_noise"] = out["gap"].rolling(window).std(ddof=0).shift(1)
        out["slope_change"] = out["slope"].diff()
        out["innovation"] = (out["slope_change"] / (out["gap_noise"] / window).where(lambda s: s > 0)).replace([np.inf, -np.inf], np.nan)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["conviction"] = (0.35 + 0.65 * (out["innovation"] / (3.0 * params.innovation_threshold)).clip(0.0, 1.0)).fillna(0.35)
        out["entry"] = (
            (out["slope"].shift(1) <= 0) & (out["slope"] > 0)
            & (out["innovation"] >= params.innovation_threshold)
            & (out["gap"] > 0) & (out["intraday"] < 0)
            & (out["intraday"] >= -0.5 * out["gap"])
            & np.isfinite(out["atr"]) & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.35, dtype=float)
        entry_decision = -1
        fixed_atr = 0.0
        stop_price = np.nan
        allocation = 0.35
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        conviction = indicators["conviction"].to_numpy(dtype=float)
        for i in range(len(data)):
            if entry_decision >= 0:
                # Observe the entry fill only on its execution bar.
                # Freeze the decision-bar ATR and the entry-open stop.
                if i == entry_decision + 1:
                    stop_price = opening[i] - params.stop_atr * fixed_atr
                invalid = not np.isfinite(stop_price) or not np.isfinite(closing[i]) or opening[i] <= 0
                if invalid or closing[i] < stop_price or i - entry_decision >= params.hold_bars:
                    entry_decision = -1
                    continue
                signal[i] = 1
                sizes[i] = allocation
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(conviction[i]):
                entry_decision = i
                fixed_atr = atr[i]
                allocation = float(np.clip(conviction[i], 0.35, 1.0))
                signal[i] = 1
                sizes[i] = allocation
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.35).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=4)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 96.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"entry": [True, False, False, False], "atr": 1.0,
                        "conviction": 0.8}, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    params = GeneratedParams()
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0]
    assert result["size"].iloc[1:3].tolist() == [0.8, 0.8]
    data.loc[index[1], "close"] = 98.0
    ind["atr"] = [1.0, 50.0, 50.0, 50.0]
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0]
    assert strategy.indicators(data.iloc[:0], params).empty
