from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeParams:
    range_window: int = 12
    percentile_window: int = 126
    entry_quantile: float = 0.80
    atr_window: int = 14
    volatility_window: int = 20
    target_volatility: float = 0.12
    breakeven_pct: float = 0.025
    trail_atr: float = 3.0
    initial_stop_atr: float = 3.0
    max_hold: int = 20


class GeneratedStrategy(BaseStrategy[RangeParams]):
    strategy_id = "gen_local-w2_1791291227"

    @classmethod
    def params_type(cls):
        return RangeParams

    @staticmethod
    def warmup_bars(params: RangeParams) -> int:
        return int(max(params.range_window + params.percentile_window + 2,
                       params.atr_window + 1, params.volatility_window))

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeParams) -> pd.DataFrame:
        if (min(params.range_window, params.percentile_window,
                params.atr_window, params.volatility_window, params.max_hold) < 1
                or not 0.0 < params.entry_quantile < 1.0
                or min(params.target_volatility, params.breakeven_pct,
                       params.trail_atr, params.initial_stop_atr) <= 0.0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        high, low, close = data["high"], data["low"], data["close"]
        out["width"] = (high - low).where((high > low) & (low > 0.0))
        out["midpoint"] = (high + low) / 2.0
        out["envelope"] = (high.rolling(params.range_window).max()
                           - low.rolling(params.range_window).min())
        out["recycling"] = (out["width"].rolling(params.range_window).sum()
                            / out["envelope"].where(out["envelope"] > 0.0))
        out["threshold"] = out["recycling"].shift(1).rolling(
            params.percentile_window).quantile(params.entry_quantile)
        out["mean_width"] = out["width"].rolling(params.range_window).mean()
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [high - low, (high - previous_close).abs(),
             (low - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["log_range_squared"] = np.log(
            high.where(high > 0.0) / low.where(low > 0.0)).pow(2)
        out["annual_volatility"] = np.sqrt(
            out["log_range_squared"].rolling(params.volatility_window).mean()
            * 252.0 / (4.0 * np.log(2.0)))
        out["risk_size"] = (params.target_volatility
                            / out["annual_volatility"].where(
                                out["annual_volatility"] > 0.0)).clip(0.01, 0.95)
        out["entry"] = (
            (out["recycling"].shift(1) > out["threshold"].shift(1))
            & (out["recycling"] < out["recycling"].shift(1))
            & (out["width"] > out["mean_width"].shift(1))
            & (out["midpoint"] > out["midpoint"].shift(1))
            & (close > out["midpoint"])
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RangeParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        risk = indicators["risk_size"].to_numpy(dtype=float)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        active = pending = armed = False
        entry_price = peak = stop = 0.0
        age = 0
        for i in range(len(data)):
            valid = (np.isfinite(prices[i]).all() and (prices[i] > 0.0).all()
                     and np.isfinite(atr[i]) and atr[i] > 0.0
                     and np.isfinite(risk[i]) and risk[i] > 0.0)
            if not valid:
                active = pending = armed = False
                continue
            if pending:
                # The preceding close requested a fill at this bar's open.
                entry_price = peak = prices[i, 0]
                stop = entry_price - params.initial_stop_atr * atr[i - 1]
                age = 0
                armed = False
                active, pending = True, False
            if active:
                age += 1
                # Only the stop known before this bar can trigger on its low.
                if prices[i, 2] <= stop or age >= params.max_hold:
                    active = False
                    continue
                peak = max(peak, prices[i, 1])
                if peak >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price, peak - params.trail_atr * atr[i])
                    # A newly raised stop breached at the close requests an exit.
                    if prices[i, 3] <= stop:
                        active = False
                        continue
                decisions[i] = 1
            elif entries[i]:
                pending = True
                decisions[i] = 1
        df["signal"] = decisions
        df["size"] = indicators["risk_size"].fillna(0.01).clip(0.01, 0.95)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RangeParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100, 100, 103, 104, 104, 104],
        "high": [101, 104, 105, 105, 105, 105],
        "low": [99, 101, 103, 100, 103, 103],
        "close": [100, 103, 104, 104, 104, 104],
        "volume": [1000] * 6,
    }, index=index)
    derived = pd.DataFrame({
        "entry": [True, False, False, False, False, False],
        "atr": [1.0, 1.0, 10.0, 10.0, 10.0, 10.0],
        "risk_size": [0.5] * 6,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert (result["size"] > 0.0).all()
    assert result.index.equals(data.index)
    prefix = GeneratedStrategy.generate_signals(
        data.iloc[:3], derived.iloc[:3], None, params).data
    pd.testing.assert_frame_equal(prefix, result.iloc[:3])
    assert GeneratedStrategy.warmup_bars(params) >= 140


if __name__ == "__main__":
    _self_check()
