from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalFractionalParams:
    fractional_order: float = 0.5
    min_snr: float = 1.0
    holding_bars: int = 7


class GeneratedStrategy(BaseStrategy[SeasonalFractionalParams]):
    strategy_id = "gen_local-w2_1791256274"

    @classmethod
    def params_type(cls):
        return SeasonalFractionalParams

    def warmup_bars(self, params: SeasonalFractionalParams) -> int:
        return 6

    def indicators(self, data: pd.DataFrame, params: SeasonalFractionalParams) -> pd.DataFrame:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if not np.isfinite(params.min_snr) or params.min_snr <= 0:
            raise ValueError("min_snr must be positive and finite")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        fractional = close.copy()
        weight = 1.0
        total_weight = 1.0
        for lag in range(1, 5):
            weight *= (lag - 1.0 - params.fractional_order) / lag
            fractional = fractional + weight * close.shift(lag)
            total_weight += weight
        # ponytail: six-tap approximation; extend memory if the warmup cap is removed.
        fractional = fractional - total_weight * close.shift(5)
        noise = close.diff().abs().rolling(5, min_periods=5).mean()
        snr = (fractional / noise.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame({
            "fractional_close": fractional,
            "noise": noise,
            "snr": snr,
            "seasonal_window": data.index.dayofweek == 2,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalFractionalParams,
    ) -> SignalFrame:
        if not isinstance(params.holding_bars, int) or not 5 <= params.holding_bars <= 10:
            raise ValueError("holding_bars must be an integer between five and ten")
        eligible = (
            indicators["seasonal_window"].fillna(False)
            & indicators["snr"].gt(params.min_snr)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for bar in range(len(data)):
            if exit_bar >= 0:
                if bar < exit_bar:
                    signal[bar] = 1
                else:
                    # Reserve a flat bar even if a fresh entry qualifies here.
                    exit_bar = -1
            elif eligible[bar]:
                signal[bar] = 1
                exit_bar = bar + params.holding_bars
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = SeasonalFractionalParams()
    index = pd.date_range("2024-01-01", periods=24, freq="B")
    close = np.arange(100.0, 124.0)
    data = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.0,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=6)
    forced = pd.DataFrame({"seasonal_window": True, "snr": 2.0}, index=index)
    result = strategy.generate_signals(data, forced, ctx, params).data
    assert result["signal"].iloc[:9].tolist() == [0] + [1] * 7 + [0]
    assert result["signal"].iloc[9:17].tolist() == [1] * 7 + [0]
    actual = strategy.indicators(data, params)
    assert actual["snr"].iloc[:5].isna().all()
    assert strategy.warmup_bars(params) <= 10
    pd.testing.assert_frame_equal(actual.iloc[:12], strategy.indicators(data.iloc[:12], params))
