from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    regime_window: int = 25
    streak_bars: int = 2
    pressure_window: int = 3
    pressure_threshold: float = 0.15
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791334707"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        if min(params.regime_window, params.streak_bars, params.pressure_window) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if params.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be 3, 4, or 5.")
        if not 0.0 <= params.pressure_threshold < 1.0:
            raise ValueError("pressure_threshold must be in [0, 1).")
        return int(max(params.regime_window + 2, params.streak_bars + 1, params.pressure_window))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lagged_return"] = out["return"].shift(1)
        out["autocorrelation"] = out["return"].rolling(params.regime_window).corr(out["lagged_return"]).replace([np.inf, -np.inf], np.nan)
        for name, condition in (("up_streak", out["return"].gt(0)), ("down_streak", out["return"].lt(0))):
            out[name] = condition.astype(int).groupby((~condition).cumsum()).cumsum().clip(upper=params.streak_bars)
        out["range"] = (data["high"] - data["low"]).where(lambda s: s > 0)
        out["close_location"] = ((2.0 * data["close"] - data["high"] - data["low"]) / out["range"]).clip(-1.0, 1.0)
        out["valid_volume"] = data["volume"].where(data["volume"].gt(0) & out["close_location"].notna())
        out["signed_volume"] = out["close_location"] * out["valid_volume"]
        out["volume_sum"] = out["valid_volume"].rolling(params.pressure_window).sum()
        out["signed_volume_sum"] = out["signed_volume"].rolling(params.pressure_window).sum()
        out["pressure"] = (out["signed_volume_sum"] / out["volume_sum"]).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        self.warmup_bars(params)
        momentum = indicators["autocorrelation"].gt(0) & indicators["up_streak"].ge(params.streak_bars)
        reversal = indicators["autocorrelation"].le(0) & indicators["down_streak"].ge(params.streak_bars)
        ready = ((momentum | reversal) & indicators["pressure"].ge(params.pressure_threshold)).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        remaining = 0
        for i in range(len(data)):
            if remaining > 0:
                remaining -= 1
                signal[i] = int(remaining > 0)
            elif ready[i]:
                remaining = params.hold_bars
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    for holding in (3, 4, 5):
        params = StrategyParams(hold_bars=holding)
        index = pd.date_range("2020-01-01", periods=12)
        data = pd.DataFrame(index=index)
        ind = pd.DataFrame({"autocorrelation": 0.25, "up_streak": 2, "down_streak": 0, "pressure": 0.5}, index=index)
        ind.loc[index[1]:, "pressure"] = -0.5
        result = strategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0] + [1] * holding + [0] * (11 - holding)
        assert result["size"].gt(0).all()
        ind.loc[:, "autocorrelation"] = np.nan
        assert strategy.generate_signals(data, ind, None, params).data["signal"].eq(0).all()
