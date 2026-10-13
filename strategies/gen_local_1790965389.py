from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    pattern_window: int = 24
    complexity_threshold: float = 1.0
    atr_window: int = 14
    trail_atr: float = 1.0


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790965389"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.pattern_window + 5, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.pattern_window < 6 or params.atr_window < 1:
            raise ValueError("Invalid indicator window")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        if not np.isfinite(params.complexity_threshold) or params.complexity_threshold <= 0:
            raise ValueError("complexity_threshold must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["gap"] = data["open"].div(out["previous_close"].where(out["previous_close"] > 0)) - 1.0
        out["gap"] = out["gap"].replace([np.inf, -np.inf], np.nan)
        out["gap_lag1"] = out["gap"].shift(1)
        out["gap_lag2"] = out["gap"].shift(2)
        a, b, c = out["gap_lag2"], out["gap_lag1"], out["gap"]
        valid = a.notna() & b.notna() & c.notna()
        # Equal gaps retain chronological order in the ordinal ranking.
        out["pattern"] = (
            4 * (a > b).astype(int) + 2 * (a > c).astype(int) + (b > c).astype(int)
        ).where(valid)
        out["entropy"] = 0.0
        out["pattern_probability"] = np.nan
        for code in (0, 1, 3, 4, 6, 7):
            frequency = out["pattern"].eq(code).astype(float).where(valid).rolling(
                params.pattern_window, min_periods=params.pattern_window
            ).mean().shift(1)
            out[f"frequency_{code}"] = frequency
            out["entropy"] -= frequency * np.log(frequency.clip(lower=1e-12))
            out["pattern_probability"] = out["pattern_probability"].where(
                out["pattern"].ne(code), frequency
            )
        out["entropy"] /= np.log(6.0)
        out["surprise"] = -np.log(
            out["pattern_probability"].clip(lower=1.0 / (params.pattern_window + 1))
        ) / np.log(6.0)
        out["complexity"] = out["entropy"] * out["surprise"]
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["candidate"] = (
            out["complexity"].ge(params.complexity_threshold)
            & out["gap"].gt(0)
            & data["close"].gt(data["open"])
            & out["atr"].gt(0)
        ).fillna(False)
        out["confirmed"] = out["candidate"] & out["candidate"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        active = False
        age = 0
        peak = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            if active:
                # Entry was ordered at the previous close; this bar is now held.
                age += 1
                if not np.isfinite(closes[i]) or not np.isfinite(atrs[i]) or atrs[i] <= 0:
                    active = False
                    continue
                peak = max(peak, closes[i])
                stop = max(stop, peak - params.trail_atr * atrs[i])
                if closes[i] <= stop or age >= 2:
                    active = False
                else:
                    decisions[i] = 1
            elif confirmed[i] and np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0:
                active = True
                age = 0
                peak = -np.inf
                stop = -np.inf
                decisions[i] = 1
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=80)
    t = np.arange(80, dtype=float)
    close = 100.0 + 0.1 * t + np.sin(t)
    data = pd.DataFrame({
        "open": close - 0.2, "high": close + 1.0,
        "low": close - 1.0, "close": close, "volume": 1000000.0,
    }, index=index)
    strategy = GeneratedStrategy()
    params = StrategyParams()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    indicators = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(
        indicators.iloc[:60], strategy.indicators(data.iloc[:60], params)
    )
    assert not indicators["confirmed"].iloc[:params.pattern_window + 4].any()
    controlled = indicators.copy()
    controlled["confirmed"] = False
    controlled.iloc[35, controlled.columns.get_loc("confirmed")] = True
    result = strategy.generate_signals(data, controlled, ctx, params).data
    assert result["signal"].iloc[35:39].tolist() == [0, 1, 1, 0]
    assert set(result["signal"].unique()) <= {0, 1}
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
