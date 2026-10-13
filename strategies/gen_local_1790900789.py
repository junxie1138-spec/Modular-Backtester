from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    dependence_window: int = 30
    rank_window: int = 80
    variance_window: int = 10
    entry_percentile: float = 0.75
    atr_window: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790900789"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(
            params.dependence_window + params.rank_window + 2,
            params.variance_window + params.rank_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.dependence_window, params.rank_window, params.variance_window, params.atr_window) < 3:
            raise ValueError("Indicator windows must be at least three bars")
        if not 0.5 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between 0.5 and 1")
        if params.trail_atr <= 0 or params.breakeven_pct <= 0 or params.max_hold_bars < 1:
            raise ValueError("Exit parameters must be positive")

        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lag_one"] = out["return"].shift(1)
        out["lag_two"] = out["return"].shift(2)
        window = params.dependence_window
        out["corr_01"] = out["return"].rolling(window).corr(out["lag_one"])
        out["corr_02"] = out["return"].rolling(window).corr(out["lag_two"])
        out["corr_12"] = out["lag_one"].rolling(window).corr(out["lag_two"])
        out["partial_denominator"] = np.sqrt(
            (1.0 - out["corr_01"].clip(-1.0, 1.0).pow(2))
            * (1.0 - out["corr_12"].clip(-1.0, 1.0).pow(2))
        )
        out["partial_lag_two"] = (
            (out["corr_02"] - out["corr_01"] * out["corr_12"])
            / out["partial_denominator"].where(out["partial_denominator"] > 1e-8)
        ).clip(-1.0, 1.0)
        out["dependence_percentile"] = out["partial_lag_two"].rolling(params.rank_window).rank(pct=True)

        out["up_semivariance"] = out["return"].clip(lower=0).pow(2).rolling(params.variance_window).sum()
        out["down_semivariance"] = out["return"].clip(upper=0).pow(2).rolling(params.variance_window).sum()
        out["total_variance"] = out["up_semivariance"] + out["down_semivariance"]
        out["variance_imbalance"] = (
            (out["up_semivariance"] - out["down_semivariance"])
            / out["total_variance"].where(out["total_variance"] > 0)
        )
        out["imbalance_percentile"] = out["variance_imbalance"].rolling(params.rank_window).rank(pct=True)

        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)

        # Positive lag-two dependence predicts the next return from today's lag-one return.
        out["dependence_agrees"] = (
            out["partial_lag_two"].gt(0)
            & out["dependence_percentile"].ge(params.entry_percentile)
            & out["lag_one"].gt(0)
        )
        out["variance_agrees"] = (
            out["variance_imbalance"].gt(0)
            & out["imbalance_percentile"].ge(params.entry_percentile)
        )
        out["entry"] = (
            out["dependence_agrees"] & out["variance_agrees"]
            & out["atr"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = False
        armed = False
        entry_price = peak = stop = 0.0
        entry_bar = 0

        for i in range(len(data)):
            # Yesterday's entry decision fills at today's open.
            if not active and i > 0 and raw[i - 1] == 1:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    armed = False
                    entry_price = peak = opens[i]
                    entry_bar = i
                    stop = entry_price - params.trail_atr * atr[i - 1]

            if active:
                # Only the stop established before this session can fire today.
                stopped = np.isfinite(lows[i]) and lows[i] <= stop
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if stopped or expired:
                    active = False
                    raw[i] = 0
                    continue

                raw[i] = 1
                if np.isfinite(highs[i]):
                    peak = max(peak, highs[i])
                    armed = armed or peak >= entry_price * (1.0 + params.breakeven_pct)
                if armed:
                    stop = max(stop, entry_price)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        stop = max(stop, peak - params.trail_atr * atr[i])
                # Updated stops become effective in the following session.
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                raw[i] = 1

        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    prices = pd.DataFrame({
        "open": [100.0] * 6,
        "high": [100.0, 100.0, 103.0, 101.0, 101.0, 101.0],
        "low": [99.0, 99.0, 101.0, 100.0, 99.5, 99.5],
        "close": [100.0, 100.0, 102.0, 100.5, 100.0, 100.0],
        "volume": [1000.0] * 6,
    }, index=index)
    derived = pd.DataFrame({"entry": [True, False, False, False, False, False], "atr": 1.0}, index=index)
    strategy = GeneratedStrategy()
    params = StrategyParams(trail_atr=2.0, breakeven_pct=0.01)
    result = strategy.generate_signals(prices, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    flat = prices.assign(open=100.0, high=100.0, low=100.0, close=100.0)
    assert not strategy.indicators(flat, params)["entry"].any()
