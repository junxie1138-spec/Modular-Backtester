from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalStrengthParams:
    frac_order: float = 0.45
    frac_window: int = 32
    impulse_lag: int = 5
    strength_window: int = 8
    percentile_window: int = 126
    entry_percentile: float = 0.80
    min_positive: int = 6
    atr_window: int = 14
    breakeven_pct: float = 0.02
    initial_stop_atr: float = 3.0
    trail_atr: float = 2.5
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[FractionalStrengthParams]):
    strategy_id = "gen_local-w2_1791271139"

    @classmethod
    def params_type(cls) -> type[FractionalStrengthParams]:
        return FractionalStrengthParams

    def warmup_bars(self, params: FractionalStrengthParams) -> int:
        return int(max(
            params.frac_window + params.impulse_lag
            + params.strength_window + params.percentile_window + 2,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: FractionalStrengthParams) -> pd.DataFrame:
        if not (
            0.0 < params.frac_order < 1.0
            and params.frac_window >= 2
            and params.impulse_lag >= 1
            and params.strength_window >= 2
            and params.percentile_window >= 2
            and 0.0 < params.entry_percentile < 1.0
            and 1 <= params.min_positive <= params.strength_window
            and params.atr_window >= 2
            and params.breakeven_pct > 0.0
            and params.initial_stop_atr > 0.0
            and params.trail_atr > 0.0
            and params.max_hold >= 1
        ):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        # Finite causal fractional difference: w_j = -w_(j-1) * (d-j+1)/j.
        weights = np.ones(params.frac_window, dtype=float)
        for j in range(1, params.frac_window):
            weights[j] = -weights[j - 1] * (params.frac_order - j + 1) / j
        fractional = close.copy()
        for j in range(1, params.frac_window):
            fractional = fractional + weights[j] * close.shift(j)
        out["fractional_close"] = fractional
        out["fractional_change"] = fractional.diff(params.impulse_lag)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["impulse"] = out["fractional_change"] / out["atr"].replace(0.0, np.nan)
        impulses = out["impulse"].rolling(
            params.strength_window, min_periods=params.strength_window
        )
        out["impulse_mean"] = impulses.mean()
        out["impulse_std"] = impulses.std(ddof=0)
        out["strength"] = out["impulse_mean"] / out["impulse_std"].replace(0.0, np.nan)
        out["threshold"] = out["strength"].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.entry_percentile).shift(1)
        positive = (out["impulse"] > 0.0).astype(float).where(out["impulse"].notna())
        out["positive_count"] = positive.rolling(
            params.strength_window, min_periods=params.strength_window
        ).sum()
        out["entry"] = (
            (out["strength"] > out["threshold"])
            & (out["strength"].shift(1) <= out["threshold"].shift(1))
            & (out["impulse_mean"] > 0.0)
            & (out["positive_count"] >= params.min_positive)
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FractionalStrengthParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        active = False
        pending = False
        armed = False
        entry_price = peak = stop = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0.0 and np.isfinite(atr[i - 1]):
                    active = True
                    armed = False
                    entry_price = peak = opens[i]
                    entry_bar = i
                    stop = entry_price - params.initial_stop_atr * atr[i - 1]
            if active:
                if not np.isfinite(closes[i]) or not np.isfinite(atr[i]) or atr[i] <= 0.0:
                    active = False
                    continue
                # Close-observed stops exit at next open, never at a synthetic stop fill.
                if closes[i] <= stop or i - entry_bar + 1 >= params.max_hold:
                    active = False
                    continue
                peak = max(peak, closes[i])
                if closes[i] >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price, peak - params.trail_atr * atr[i])
                    if closes[i] <= stop:
                        active = False
                        continue
                decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0 and np.isfinite(atr[i]) and atr[i] > 0.0:
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0, 101.0, 104.0, 103.0, 105.0, 102.0],
        "close": [100.0, 104.0, 103.0, 105.0, 102.0, 110.0],
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [True, False, False, False, False, False],
        "atr": [1.0, 1.0, 10.0, 1.0, 1.0, 1.0],
    }, index=index)
    strategy = GeneratedStrategy()
    params = FractionalStrengthParams()
    result = strategy.generate_signals(data, indicators, None, params).data
    # Rising ATR cannot lower the armed stop; the exit decision fills one bar later.
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    params.max_hold = 2
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
