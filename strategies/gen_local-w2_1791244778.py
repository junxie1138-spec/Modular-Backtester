from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapParams:
    atr_window: int = 14
    gap_window: int = 20
    percentile_window: int = 126
    shock_quantile: float = 0.8
    stop_atr: float = 2.5
    hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[GapParams]):
    strategy_id = "gen_local-w2_1791244778"

    @classmethod
    def params_type(cls):
        return GapParams

    def warmup_bars(self, params: GapParams) -> int:
        return int(max(params.atr_window, params.gap_window) + params.percentile_window + 2)

    def indicators(self, data: pd.DataFrame, params: GapParams) -> pd.DataFrame:
        if min(params.atr_window, params.gap_window) < 2 or params.percentile_window < 2:
            raise ValueError("Lookback windows must be at least two.")
        if not 0.5 < params.shock_quantile < 1.0:
            raise ValueError("shock_quantile must be between 0.5 and 1.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0 or params.hold_bars < 1:
            raise ValueError("Stop distance and holding period must be positive.")
        previous = data["close"].shift(1).replace(0.0, np.nan)
        gap = data["open"].div(previous).sub(1.0)
        gap_sigma = gap.rolling(params.gap_window).std().shift(1).replace(0.0, np.nan)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean()
        gap_score = gap.abs().div(gap_sigma)
        quiet_score = (data["high"] - data["low"]).div(atr.shift(1).replace(0.0, np.nan))
        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["gap_sigma"] = gap_sigma
        out["true_range"] = true_range
        out["atr"] = atr
        out["gap_score"] = gap_score
        out["quiet_score"] = quiet_score
        out["gap_cutoff"] = gap_score.shift(1).rolling(params.percentile_window).quantile(params.shock_quantile)
        out["quiet_cutoff"] = quiet_score.shift(1).rolling(params.percentile_window).quantile(1.0 - params.shock_quantile)
        out["acceptance"] = (data["close"] - data["open"]) * gap
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapParams,
    ) -> SignalFrame:
        required = ["gap", "atr", "gap_score", "quiet_score", "gap_cutoff", "quiet_cutoff", "acceptance"]
        ready = np.isfinite(indicators[required]).all(axis=1) & indicators["atr"].gt(0)
        accepted = (
            ready
            & indicators["gap_score"].gt(indicators["gap_cutoff"])
            & indicators["quiet_score"].lt(indicators["quiet_cutoff"])
            & indicators["acceptance"].gt(0)
        )
        entries = np.sign(indicators["gap"]).where(accepted, 0).fillna(0).to_numpy(dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        valid = ready.to_numpy(dtype=bool)
        desired = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        high_water, trailing_floor, short_ceiling = np.nan, -np.inf, np.inf
        for i in range(len(data)):
            if position:
                age += 1
                if not valid[i] or not np.isfinite(closes[i]) or not np.isfinite(opens[i]):
                    position = 0
                else:
                    if age == 1:
                        # Entry fills at this bar's open; track only post-entry closes.
                        high_water = closes[i]
                        trailing_floor = high_water - params.stop_atr * atrs[i]
                        short_ceiling = opens[i] + params.stop_atr * atrs[i - 1]
                    else:
                        high_water = max(high_water, closes[i])
                        trailing_floor = max(trailing_floor, high_water - params.stop_atr * atrs[i])
                    # The rolling-high exit applies to both directions.
                    # Shorts also exit on an adverse move above their entry-open limit.
                    if (
                        closes[i] <= trailing_floor
                        or (position < 0 and closes[i] >= short_ceiling)
                        or age >= params.hold_bars
                    ):
                        position = 0
                # An exit leaves one flat target bar before another entry.
            elif valid[i] and np.isfinite(closes[i]) and np.isfinite(opens[i]) and entries[i]:
                position = int(entries[i])
                age = 0
                high_water, trailing_floor, short_ceiling = np.nan, -np.inf, np.inf
            desired[i] = position
        df = pd.DataFrame({"signal": desired, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 106.0, "low": 90.0,
                         "close": [100., 102., 105., 104., 100., 99., 98.],
                         "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"gap": 0.01, "atr": [1., 1., 1., 5., 1., 1., 1.],
                        "gap_score": [3., 0., 0., 0., 0., 0., 0.],
                        "quiet_score": 0.1, "gap_cutoff": 2.0,
                        "quiet_cutoff": 0.2, "acceptance": 1.0}, index=index)
    params = GapParams(stop_atr=2.0)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    data["close"] = [100., 99., 98., 97., 95., 94., 93.]
    ind["gap"], ind["atr"] = -0.01, 1.0
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, -1, -1, -1, 0, 0, 0]
    ind["gap_cutoff"] = np.nan
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
