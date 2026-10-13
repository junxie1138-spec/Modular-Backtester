from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRetestParams:
    atr_window: int = 14
    volume_window: int = 20
    percentile_window: int = 126
    gap_quantile: float = 0.75
    volume_quantile: float = 0.70
    quiet_quantile: float = 0.40
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[GapRetestParams]):
    strategy_id = "gen_local-w2_1791323714"

    @classmethod
    def params_type(cls):
        return GapRetestParams

    def warmup_bars(self, params: GapRetestParams) -> int:
        return max(params.atr_window, params.volume_window) + params.percentile_window + 3

    def indicators(self, data: pd.DataFrame, params: GapRetestParams) -> pd.DataFrame:
        if min(params.atr_window, params.volume_window, params.percentile_window, params.max_hold) < 1:
            raise ValueError("Windows and holding limit must be positive.")
        if not all(0.0 < q < 1.0 for q in (params.gap_quantile, params.volume_quantile, params.quiet_quantile)):
            raise ValueError("Quantiles must lie strictly between zero and one.")
        if params.quiet_quantile >= params.volume_quantile or params.trail_atr <= 0 or params.breakeven_pct <= 0:
            raise ValueError("Require quiet < volume quantile and positive stop parameters.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["gap"] = (data["open"] - out["previous_close"]) / out["atr"].shift(1).replace(0, np.nan)
        out["volume_baseline"] = data["volume"].shift(1).rolling(params.volume_window).median()
        out["relative_volume"] = data["volume"] / out["volume_baseline"].replace(0, np.nan)
        history = params.percentile_window
        out["gap_threshold"] = out["gap"].abs().shift(1).rolling(history).quantile(params.gap_quantile)
        out["volume_threshold"] = out["relative_volume"].shift(1).rolling(history).quantile(params.volume_quantile)
        out["quiet_threshold"] = out["relative_volume"].shift(1).rolling(history).quantile(params.quiet_quantile)
        out["spike"] = (
            (out["gap"] > out["gap_threshold"])
            & (out["relative_volume"] > out["volume_threshold"])
            & (data["close"] > data["open"])
        ).fillna(False)
        # One quiet session must test the spike's opening price without filling its gap.
        out["retest"] = (
            out["spike"].shift(1, fill_value=False)
            & (data["low"] <= data["open"].shift(1))
            & (data["low"] > data["close"].shift(2))
            & (data["close"] >= data["open"].shift(1))
            & (out["relative_volume"] < out["quiet_threshold"])
        ).fillna(False)
        out["entry"] = (
            out["retest"].shift(1, fill_value=False)
            & (data["close"] > data["high"].shift(1))
            & (data["low"] > data["close"].shift(3))
            & (out["relative_volume"] > out["volume_threshold"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapRetestParams) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.95}, index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        holding = pending = armed = False
        entry = peak = stop = 0.0
        age = 0
        for i, (opening, high, low, close) in enumerate(prices):
            valid = np.isfinite([opening, high, low, close, atr[i]]).all() and min(opening, low, atr[i]) > 0
            if pending:
                pending = False
                if i > 0 and valid and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    # This open is known only now; yesterday's decision never uses it.
                    entry = peak = opening
                    stop = entry - params.trail_atr * atr[i - 1]
                    holding, armed, age = True, False, 0
            if holding:
                age += 1
                # Test the existing stop before ratcheting at this close.
                if not valid or low <= stop or age >= params.max_hold:
                    holding = False
                    continue
                peak = max(peak, high)
                armed = armed or high >= entry * (1.0 + params.breakeven_pct)
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    if close <= stop:
                        holding = False
                        continue
                decisions[i] = 1
            elif valid and entries[i]:
                pending = True
                decisions[i] = 1
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=9)
    close = np.array([99., 101., 103., 104., 101., 104., 105., 101., 101.])
    opening = np.array([99., 100., 101., 103., 104., 101., 104., 105., 101.])
    data = pd.DataFrame({"open": opening, "high": np.maximum(opening, close) + 0.5,
                         "low": np.minimum(opening, close) - 0.5,
                         "close": close, "volume": 1000.}, index=index)
    params = GapRetestParams()
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    synthetic = pd.DataFrame({"atr": [1., 1., 1., 5., 5., 5., 5., 5., 5.],
                              "entry": [True] + [False] * 8}, index=index)
    result = strategy.generate_signals(data, synthetic, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0]
    params.max_hold = 2
    capped = strategy.generate_signals(data, synthetic, ctx, params).data
    assert capped["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0]
    params.max_hold = 8
    prefix = strategy.generate_signals(data.iloc[:4], synthetic.iloc[:4], ctx, params).data
    assert prefix.equals(result.iloc[:4])
    warmup = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert warmup["signal"].eq(0).all() and warmup["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
