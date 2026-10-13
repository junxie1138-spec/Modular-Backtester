from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    area_window: int = 5
    area_threshold: float = 0.08
    pressure_window: int = 6
    pressure_jump: float = 0.8
    atr_window: int = 14
    trail_atr: float = 0.75
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791262201"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(params.area_window + 2, params.pressure_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if (params.area_window < 2 or params.pressure_window < 1
                or params.atr_window < 1 or params.hold_bars < 1
                or params.area_threshold <= 0 or params.pressure_jump <= 0
                or params.trail_atr <= 0):
            raise ValueError("Lookbacks and thresholds must be positive; area_window >= 2.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["return"] = close.pct_change(fill_method=None)
        out["range"] = data["high"] - data["low"]
        out["true_range"] = pd.concat(
            [out["range"], (data["high"] - close.shift(1)).abs(),
             (data["low"] - close.shift(1)).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["pressure"] = (
            (2 * close - data["high"] - data["low"])
            / out["range"].replace(0, np.nan)
        ).clip(-1, 1)
        out["pressure_baseline"] = out["pressure"].shift(1).rolling(
            params.pressure_window
        ).mean()
        out["pressure_change"] = out["pressure"] - out["pressure_baseline"]
        length = params.area_window
        # Antisymmetric level-two signature of (unit time, cumulative returns).
        out["area_numerator"] = sum(
            (length - 1 - 2 * lag) * out["return"].shift(lag)
            for lag in range(length)
        )
        out["area_scale"] = (
            2 * (length - 1) * out["return"].abs().rolling(length).sum()
        )
        out["area"] = out["area_numerator"] / out["area_scale"].clip(lower=1e-12)
        out["previous_area"] = out["area"].shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SignatureParams,
    ) -> SignalFrame:
        ready = indicators[["area", "previous_area", "pressure_change", "atr"]].notna().all(axis=1)
        ready &= indicators["atr"].gt(0) & data["close"].notna()
        long_entry = (ready & indicators["area"].gt(params.area_threshold)
                      & indicators["previous_area"].le(0)
                      & indicators["pressure_change"].gt(params.pressure_jump))
        short_entry = (ready & indicators["area"].lt(-params.area_threshold)
                       & indicators["previous_area"].ge(0)
                       & indicators["pressure_change"].lt(-params.pressure_jump))
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        decisions = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        high_water, stop = -np.inf, -np.inf
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        for i in range(len(data)):
            if side:
                age += 1
                if not np.isfinite(closes[i]) or not np.isfinite(atrs[i]) or atrs[i] <= 0:
                    side = 0
                else:
                    # First held close is on the bar after the entry decision.
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.trail_atr * atrs[i])
                    # The specified high-close drawdown exit also harvests short profits.
                    if closes[i] < stop or age >= params.hold_bars:
                        side = 0
            elif entries[i]:
                side, age = int(entries[i]), 0
                high_water, stop = -np.inf, -np.inf
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = SignatureParams(hold_bars=4, trail_atr=1.0)
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100, 103, 102.5, 101.5, 100, 99, 98]}, index=index)
    ind = pd.DataFrame({
        "area": [0.2, 0, 0, 0, -0.2, 0, 0],
        "previous_area": [0, 0, 0, 0, 0, 0, 0],
        "pressure_change": [1, 0, 0, 0, -1, 0, 0],
        "atr": [1, 1, 4, 4, 1, 1, 1],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1]
    veto = ind.copy()
    veto["pressure_change"] = 0.0
    assert strategy.generate_signals(data, veto, ctx, params).data["signal"].eq(0).all()
    params.hold_bars = 2
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1]


if __name__ == "__main__":
    _self_check()
