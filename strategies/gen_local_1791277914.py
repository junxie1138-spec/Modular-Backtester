from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RepricingParams:
    recent_window: int = 5
    baseline_window: int = 20
    percentile_window: int = 126
    contraction_percentile: float = 0.15
    displacement_percentile: float = 0.65
    atr_window: int = 14
    trail_atr: float = 3.0
    holding_bars: int = 20


class GeneratedStrategy(BaseStrategy[RepricingParams]):
    strategy_id = "gen_local_1791277914"

    @classmethod
    def params_type(cls) -> type[RepricingParams]:
        return RepricingParams

    def warmup_bars(self, params: RepricingParams) -> int:
        return int(max(params.recent_window + params.baseline_window + 1, params.recent_window + params.atr_window + 1) + params.percentile_window)

    def indicators(self, data: pd.DataFrame, params: RepricingParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["log_return"] = np.log(close.where(close > 0)).diff()
        out["recent_vol"] = out["log_return"].rolling(params.recent_window).std(ddof=0)
        out["baseline_vol"] = out["log_return"].rolling(params.baseline_window).std(ddof=0).shift(params.recent_window)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([(data["high"] - data["low"]), (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["baseline_atr"] = out["atr"].shift(params.recent_window)
        out["recent_median"] = close.rolling(params.recent_window).median()
        out["baseline_median"] = close.rolling(params.baseline_window).median().shift(params.recent_window)
        ratio = out["recent_vol"] / out["baseline_vol"].where(out["baseline_vol"] > 0)
        out["vol_change"] = np.log(ratio.where(ratio > 0))
        out["displacement"] = (out["recent_median"] - out["baseline_median"]) / out["baseline_atr"].where(out["baseline_atr"] > 0)
        out["contraction_threshold"] = out["vol_change"].rolling(params.percentile_window).quantile(params.contraction_percentile).shift(1)
        out["displacement_threshold"] = out["displacement"].rolling(params.percentile_window).quantile(params.displacement_percentile).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RepricingParams) -> SignalFrame:
        contraction = indicators["vol_change"] < indicators["contraction_threshold"]
        crossing = contraction & ~contraction.shift(1, fill_value=False)
        retained = (indicators["displacement"] > indicators["displacement_threshold"]) & (indicators["displacement"] > 0)
        required = ["vol_change", "contraction_threshold", "displacement", "displacement_threshold", "atr"]
        ready = np.isfinite(indicators[required].to_numpy()).all(axis=1)
        entries = (crossing & retained).to_numpy() & ready
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        highest_close = 0.0
        stop = 0.0
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                active = False
                age = 0
                continue
            if active:
                age += 1
                highest_close = max(highest_close, close)
                stop = max(stop, highest_close - params.trail_atr * atr)
                if close <= stop or age >= params.holding_bars:
                    active = False
                    age = 0
                    continue
            elif entries[i]:
                active = True
                age = 0
                highest_close = close
                stop = close - params.trail_atr * atr
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    close = np.array([100.0, 100.0, 110.0, 105.0, 103.0, 103.0])
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"vol_change": [0.0, -2.0, 0.0, 0.0, 0.0, 0.0], "contraction_threshold": -1.0, "displacement": 2.0, "displacement_threshold": 1.0, "atr": [2.0, 2.0, 2.0, 5.0, 5.0, 5.0]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, RepricingParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0]
    result = strategy.generate_signals(data, indicators, None, RepricingParams(holding_bars=2)).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    blank = strategy.indicators(data, RepricingParams())
    assert strategy.generate_signals(data, blank, None, RepricingParams()).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
