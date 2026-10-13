from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationEchoParams:
    correlation_window: int = 60
    acceleration_threshold: float = 0.6
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[AccelerationEchoParams]):
    strategy_id = "gen_local_1791342601"

    @classmethod
    def params_type(cls) -> type[AccelerationEchoParams]:
        return AccelerationEchoParams

    def warmup_bars(self, params: AccelerationEchoParams) -> int:
        if params.correlation_window < 5 or params.holding_bars < 1:
            raise ValueError("Window must be at least five and holding period positive.")
        if not np.isfinite(params.acceleration_threshold) or params.acceleration_threshold <= 0:
            raise ValueError("Acceleration threshold must be finite and positive.")
        return max(200, params.correlation_window + 4)

    def indicators(self, data: pd.DataFrame, params: AccelerationEchoParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        window = params.correlation_window
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["roc"] = close.pct_change(fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["acceleration_lag1"] = out["acceleration"].shift(1)
        out["acceleration_lag2"] = out["acceleration"].shift(2)
        rolling = out["acceleration"].rolling(window, min_periods=window)
        out["autocorr1"] = rolling.corr(out["acceleration_lag1"])
        out["autocorr2"] = rolling.corr(out["acceleration_lag2"])
        out["acceleration_noise"] = rolling.std(ddof=0)
        out["acceleration_score"] = out["acceleration"].div(
            out["acceleration_noise"].where(out["acceleration_noise"] > 0)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AccelerationEchoParams,
    ) -> SignalFrame:
        warmup = self.warmup_bars(params)
        ready = indicators.notna().all(axis=1)
        candidate = (
            ready
            & (data["close"] > indicators["ma200"])
            & (indicators["autocorr2"] > 0.15)
            & ((indicators["autocorr2"] - indicators["autocorr1"]) > 0.25)
            & (indicators["acceleration_score"] >= params.acceleration_threshold)
        ).fillna(False).to_numpy(dtype=bool)
        candidate[:warmup - 1] = False
        signal = np.zeros(len(data), dtype=np.int8)
        entry_bar = -1
        for i in range(len(data)):
            if entry_bar >= 0:
                if i - entry_bar < params.holding_bars:
                    signal[i] = 1
                else:
                    # The expiry bar stays flat even if another entry qualifies.
                    entry_bar = -1
            elif candidate[i]:
                entry_bar = i
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = object.__new__(GeneratedStrategy)
    params = AccelerationEchoParams()
    index = pd.date_range("2020-01-01", periods=250, freq="B")
    data = pd.DataFrame(
        {"open": 110.0, "high": 111.0, "low": 109.0, "close": 110.0, "volume": 1000.0},
        index=index,
    )
    actual = strategy.indicators(data, params)
    assert strategy.generate_signals(data, actual, None, params).data["signal"].eq(0).all()
    controlled = pd.DataFrame(
        {"ma200": 100.0, "autocorr1": -0.3, "autocorr2": 0.3, "acceleration_score": 1.0},
        index=index,
    )
    first_fill = strategy.warmup_bars(params)
    # Regime reversal after entry must not shorten the fixed holding period.
    controlled.iloc[first_fill:, controlled.columns.get_loc("ma200")] = 120.0
    result = strategy.generate_signals(data, controlled, None, params).data
    assert result["signal"].iloc[:first_fill].eq(0).all()
    assert result["signal"].iloc[first_fill:first_fill + params.holding_bars].eq(1).all()
    assert result["signal"].iloc[first_fill + params.holding_bars:].eq(0).all()
    assert result["size"].gt(0).all()
