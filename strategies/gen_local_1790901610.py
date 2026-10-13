from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnSizeParams:
    lookback: int = 12
    percentile_window: int = 126
    entry_quantile: float = 0.8
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[ReturnSizeParams]):
    strategy_id = "gen_local_1790901610"

    @classmethod
    def params_type(cls):
        return ReturnSizeParams

    def warmup_bars(self, params: ReturnSizeParams) -> int:
        return int(params.lookback + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: ReturnSizeParams) -> pd.DataFrame:
        if params.lookback < 2 or params.percentile_window < 2:
            raise ValueError("Lookback windows must be at least two.")
        if not 0.0 < params.entry_quantile < 1.0 or params.hold_bars < 1:
            raise ValueError("Invalid entry quantile or holding period.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        r = out["return"]
        window = params.lookback
        out["up_count"] = (r > 0).astype(float).where(r.notna()).rolling(window).sum()
        out["down_count"] = (r < 0).astype(float).where(r.notna()).rolling(window).sum()
        out["up_mean"] = r.clip(lower=0).rolling(window).sum() / out["up_count"].replace(0, np.nan)
        out["down_mean"] = (-r.clip(upper=0)).rolling(window).sum() / out["down_count"].replace(0, np.nan)
        scale = (out["up_mean"] + out["down_mean"]).replace(0, np.nan)
        out["dominance"] = (out["up_mean"] - out["down_mean"]) / scale
        count = (out["up_count"] + out["down_count"]).replace(0, np.nan)
        out["count_imbalance"] = (out["up_count"] - out["down_count"]).abs() / count
        out["net_return"] = r.rolling(window).sum()
        history = params.percentile_window
        out["dominance_threshold"] = out["dominance"].abs().shift(1).rolling(history).quantile(params.entry_quantile)
        out["balance_threshold"] = out["count_imbalance"].shift(1).rolling(history).quantile(0.5)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ReturnSizeParams,
    ) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("Holding period must be positive.")
        eligible = (
            (indicators["dominance"].abs() > indicators["dominance_threshold"])
            & (indicators["count_imbalance"] <= indicators["balance_threshold"])
            & (indicators["dominance"] * indicators["net_return"] > 0)
        ).fillna(False)
        candidates = np.where(eligible, np.sign(indicators["dominance"].fillna(0)), 0).astype(int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i, candidate in enumerate(candidates):
            if position != 0:
                if i - entry_bar >= params.hold_bars:
                    position = 0
            elif candidate != 0:
                position = int(candidate)
                entry_bar = i
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = ReturnSizeParams(lookback=2, percentile_window=2, hold_bars=3)
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    artificial = pd.DataFrame({
        "dominance": [0.8, -0.8, -0.8, -0.8, -0.8, 0.8, 0.8, 0.8, 0.8],
        "dominance_threshold": 0.5,
        "count_imbalance": 0.0,
        "balance_threshold": 0.2,
        "net_return": [0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 0.1],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, artificial, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0]
    assert (result["size"] > 0).all()
    actual = strategy.indicators(data, params)
    assert strategy.generate_signals(data, actual, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
