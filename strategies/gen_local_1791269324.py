from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    streak_len: int = 3
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791269324"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(60, int(params.streak_len) + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.streak_len < 1 or int(params.streak_len) != params.streak_len:
            raise ValueError("streak_len must be a positive integer")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        volume = data["volume"]
        change = close.diff()
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = true_range.rolling(14, min_periods=14).mean()
        out["mean60"] = close.rolling(60, min_periods=60).mean()
        out["displacement"] = close.diff(20)
        out["path20"] = change.abs().rolling(20, min_periods=20).sum()
        out["efficiency"] = (
            out["displacement"].abs() / out["path20"].replace(0.0, np.nan)
        )
        out["change"] = change
        positive_volume = (volume > 0) & (volume.shift(1) > 0)
        drying = (change < 0) & (volume < volume.shift(1)) & positive_volume
        escalating = (change < 0) & (volume > volume.shift(1)) & positive_volume
        out["drying_streak"] = drying.astype(int).groupby((~drying).cumsum()).cumsum()
        out["escalating_streak"] = escalating.astype(int).groupby((~escalating).cumsum()).cumsum()
        out["trend_regime"] = (
            (out["efficiency"] >= 0.35)
            & (out["displacement"] > 0)
            & (close > out["mean60"])
        )
        out["range_regime"] = out["efficiency"] <= 0.25
        ready = (
            np.isfinite(close) & (close > 0)
            & np.isfinite(out["atr"]) & (out["atr"] > 0)
            & np.isfinite(out["mean60"])
            & np.isfinite(out["efficiency"])
        )
        out["entry"] = ready & (change > 0) & (
            (out["trend_regime"] & (out["drying_streak"].shift(1) >= params.streak_len))
            | (out["range_regime"] & (out["escalating_streak"].shift(1) >= params.streak_len))
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = stop = -np.inf
        first_ready = self.warmup_bars(params) - 1
        for i in range(len(data)):
            valid = np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                age += 1
                if not valid:
                    active = False
                else:
                    # Entry fills on this bar when age first becomes one.
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    if closes[i] <= stop or age >= 8:
                        active = False
            elif i >= first_ready and valid and entries[i]:
                active = True
                age = 0
                high_water = stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params(streak_len=3, trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=66, freq="B")
    close = np.full(66, 100.0)
    close[60:] = [102.0, 104.0, 102.0, 100.0, 99.0, 99.0]
    data = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": 1.0, "entry": False}, index=index)
    indicators.iloc[59, indicators.columns.get_loc("entry")] = True
    indicators.iloc[61:, indicators.columns.get_loc("atr")] = 10.0
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=60)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[59:66].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert (result["size"] > 0).all()
    indicators["atr"] = 100.0
    indicators["entry"] = False
    indicators.iloc[55, indicators.columns.get_loc("entry")] = True
    assert not strategy.generate_signals(data, indicators, ctx, params).data["signal"].any()
    indicators.iloc[55, indicators.columns.get_loc("entry")] = False
    indicators.iloc[59, indicators.columns.get_loc("entry")] = True
    extended = pd.concat([data, data.iloc[-1:].set_axis([index[-1] + pd.offsets.BDay()])])
    extended_indicators = indicators.reindex(extended.index)
    extended_indicators["atr"] = 100.0
    extended_indicators["entry"] = extended_indicators["entry"].fillna(False)
    held = strategy.generate_signals(extended, extended_indicators, ctx, params).data["signal"]
    assert held.iloc[60:].tolist() == [1] * 7
    flat = strategy.indicators(data.iloc[:30], params)
    assert not flat["entry"].any()


if __name__ == "__main__":
    _self_check()
