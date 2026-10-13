from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapComplexityParams:
    entropy_ceiling: float = 0.78
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[GapComplexityParams]):
    strategy_id = "gen_local_1790858126"

    @classmethod
    def params_type(cls):
        return GapComplexityParams

    def warmup_bars(self, params: GapComplexityParams) -> int:
        if not np.isfinite(params.entropy_ceiling) or not 0 < params.entropy_ceiling < 1:
            raise ValueError("entropy_ceiling must be between zero and one")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive")
        return 26

    def indicators(self, data: pd.DataFrame, params: GapComplexityParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1).where(lambda s: s > 0)
        out["gap"] = (data["open"] / previous - 1).replace([np.inf, -np.inf], np.nan)
        a, b, c = out["gap"].shift(2), out["gap"].shift(1), out["gap"]
        valid = a.notna() & b.notna() & c.notna()
        # Stable chronological ordering resolves ties without future data.
        out["word"] = (
            4 * a.gt(b).astype(int) + 2 * a.gt(c).astype(int) + b.gt(c).astype(int)
        ).astype(float).where(valid)
        out["entropy"] = 0.0
        for word in (0, 1, 3, 4, 6, 7):
            probability = out["word"].eq(word).astype(float).where(valid).rolling(18, min_periods=18).mean()
            out["prob_" + str(word)] = probability
            out["entropy"] -= probability * np.log(probability.clip(lower=1e-12)) / np.log(6.0)
        out["past_entropy"] = out["entropy"].shift(5)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["setup"] = (
            out["entropy"].le(params.entropy_ceiling)
            & out["past_entropy"].ge(0.80)
            & (out["past_entropy"] - out["entropy"]).ge(0.12)
            & a.lt(b) & b.lt(c) & c.gt(0)
            & data["close"].gt(data["open"])
            & out["atr"].gt(0)
        ).fillna(False)
        out["entry"] = out["setup"] & ~out["setup"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapComplexityParams,
    ) -> SignalFrame:
        warmup = self.warmup_bars(params)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = pending = armed = False
        entry = stop = peak = 0.0
        age = 0
        for i, (open_price, high, low, close) in enumerate(prices):
            finite = np.isfinite(prices[i]).all() and np.isfinite(atr[i]) and atr[i] > 0
            if pending:
                pending = False
                if not finite or open_price <= 0:
                    continue
                # Next-open reference includes the config's five-basis-point buy slippage.
                entry = open_price * 1.0005
                peak = entry
                stop = entry - params.trail_atr * atr[i - 1]
                age = 0
                armed = False
                active = True
            if active:
                age += 1
                if age >= 10:
                    active = False
                    continue
                if not finite:
                    signals[i] = 1
                    continue
                # Test the previously established stop before using this bar's high.
                if low <= stop:
                    active = False
                    continue
                peak = max(peak, high)
                armed = armed or peak >= entry * 1.01
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    # A newly raised stop is evaluated at the close, avoiding high/low ordering assumptions.
                    if close <= stop:
                        active = False
                        continue
                signals[i] = 1
            elif i >= warmup and finite and entries[i]:
                signals[i] = 1
                pending = True
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapComplexityParams()
    index = pd.bdate_range("2020-01-01", periods=65)
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=26)
    forced = pd.DataFrame({"entry": False, "atr": 1.0}, index=index)
    forced.iloc[26, 0] = True
    data.iloc[27, :4] = [100.0, 102.0, 99.8, 101.5]
    data.iloc[28, :4] = [101.5, 105.0, 101.0, 104.0]
    data.iloc[29, :4] = [104.0, 104.0, 102.8, 103.0]
    forced.iloc[29, 1] = 4.0
    data.iloc[30, :4] = [103.0, 104.0, 102.0, 103.0]
    result = strategy.generate_signals(data, forced, ctx, params).data
    expected = np.zeros(len(data), dtype=int)
    expected[27:31] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    # No activation: ten open-to-open sessions before the time exit.
    flat = data.copy()
    flat.loc[:, ["open", "high", "low", "close"]] = [100.0, 100.5, 99.5, 100.0]
    timed = strategy.generate_signals(flat, forced, ctx, params).data
    expected[:] = 0
    expected[27:37] = 1
    assert np.array_equal(timed["signal"].to_numpy(), expected)
    derived = strategy.indicators(data, params)
    changed = data.copy()
    changed.iloc[45:, :4] *= 1.2
    pd.testing.assert_frame_equal(derived.iloc[:45], strategy.indicators(changed, params).iloc[:45])
    pd.testing.assert_frame_equal(
        result.iloc[:45], strategy.generate_signals(changed, forced, ctx, params).data.iloc[:45]
    )
    assert result["size"].gt(0).all() and result["signal"].isin([0, 1]).all()
    empty = data.iloc[:0]
    assert strategy.generate_signals(empty, strategy.indicators(empty, params), ctx, params).data.empty


if __name__ == "__main__":
    _self_check()
