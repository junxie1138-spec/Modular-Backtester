from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PulseParams:
    return_window: int = 3
    decision_day: int = 7
    score_threshold: float = 0.0


class GeneratedStrategy(BaseStrategy[PulseParams]):
    strategy_id = "gen_local-w2_1791246183"

    @classmethod
    def params_type(cls):
        return PulseParams

    def warmup_bars(self, params: PulseParams) -> int:
        return max(6, params.return_window) + 1

    def indicators(self, data: pd.DataFrame, params: PulseParams) -> pd.DataFrame:
        if not 1 <= params.return_window <= 9:
            raise ValueError("return_window must be between 1 and 9")
        if not 1 <= params.decision_day <= 28:
            raise ValueError("decision_day must be between 1 and 28")
        if not np.isfinite(params.score_threshold):
            raise ValueError("score_threshold must be finite")
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        pulse = returns.rolling(params.return_window, min_periods=params.return_window).sum()
        volatility = returns.rolling(6, min_periods=6).std(ddof=0)
        scale = volatility.clip(lower=1e-8) * np.sqrt(params.return_window)
        score = (pulse / scale).replace([np.inf, -np.inf], np.nan)
        day = pd.Series(data.index.day, index=data.index)
        month = pd.Series(data.index.year * 12 + data.index.month, index=data.index)
        event = day.ge(params.decision_day) & (
            day.shift(1).lt(params.decision_day) | month.ne(month.shift(1))
        )
        ready = pulse.notna() & volatility.notna() & score.notna()
        sampled = score.gt(params.score_threshold).astype(float).where(event & ready)
        condition = sampled.ffill().fillna(0.0).astype(bool)
        return pd.DataFrame({
            "daily_return": returns.fillna(0.0),
            "pulse_return": pulse.fillna(0.0),
            "daily_volatility": volatility.fillna(0.0),
            "pulse_score": score.fillna(0.0),
            "calendar_event": event,
            "ready": ready,
            "seasonal_condition": condition,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PulseParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # A negative monthly verdict flips the carried entry condition and exits.
        df["signal"] = indicators["seasonal_condition"].fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.bdate_range("2020-01-02", periods=70)
    steps = np.where(np.arange(70) < 30, 1.002, 0.998)
    close = 100.0 * np.cumprod(steps)
    data = pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 1000.0}, index=index)
    strategy, params = GeneratedStrategy(), PulseParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=7)
    indicators = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    expected = indicators["seasonal_condition"].astype(int).shift(1).fillna(0).astype(int)
    assert signals["signal"].equals(expected.rename("signal"))
    assert signals["signal"].max() == 1 and signals["signal"].iloc[-1] == 0
    assert signals["signal"].isin([0, 1]).all() and signals["size"].gt(0).all()
    assert strategy.warmup_bars(params) <= 10
    prefix = strategy.indicators(data.iloc[:45], params)
    assert prefix.equals(indicators.iloc[:45])


if __name__ == "__main__":
    _self_check()
