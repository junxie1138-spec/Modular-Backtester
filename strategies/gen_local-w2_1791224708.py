from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    volume_window: int = 20
    atr_window: int = 14
    flow_window: int = 5
    vol_window: int = 60
    volume_multiple: float = 1.5
    move_atr: float = 0.55
    quiet_ratio: float = 0.8
    target_vol: float = 0.12
    arm_pct: float = 0.025
    trail_atr: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791224708"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.volume_window, params.atr_window,
                   params.flow_window, params.vol_window) + 1

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        previous_close = close.shift(1)
        returns = close.pct_change()
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)

        out = pd.DataFrame(index=data.index)
        out["return"] = returns.fillna(0.0)
        out["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean().fillna(0.0)
        out["volume_base"] = data["volume"].rolling(params.volume_window, min_periods=params.volume_window).mean().shift(1).fillna(0.0)
        out["flow"] = returns.rolling(params.flow_window, min_periods=params.flow_window).sum().fillna(0.0)
        out["fast_vol"] = returns.rolling(params.flow_window, min_periods=params.flow_window).std().fillna(0.0)
        out["slow_vol"] = returns.rolling(params.vol_window, min_periods=params.vol_window).std().fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        previous_close = data["close"].shift(1)
        volume_move = (
            (indicators["atr"] > 0)
            & (indicators["volume_base"] > 0)
            & (indicators["return"] > params.move_atr * indicators["atr"] / previous_close)
            & (data["volume"] > params.volume_multiple * indicators["volume_base"])
        )
        quiet_flow = (
            (indicators["slow_vol"] > 0)
            & (indicators["flow"] > 0)
            & (indicators["fast_vol"] < params.quiet_ratio * indicators["slow_vol"])
        )
        enter = (volume_move & quiet_flow).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_price = 0.0
        stop = 0.0
        armed = False
        held = 0

        for i in range(len(data)):
            if i and raw[i - 1] == 1 and not active:
                active = True
                entry_price = opens[i]
                stop = 0.0
                armed = False
                held = 0
            elif i and raw[i - 1] == 0 and active:
                active = False

            if active:
                held += 1
                if highs[i] >= entry_price * (1.0 + params.arm_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price, highs[i] - params.trail_atr * atr[i])
                raw[i] = 0 if (armed and closes[i] <= stop) or held >= params.max_hold else 1
            elif enter[i]:
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        annual_vol = indicators["slow_vol"] * np.sqrt(252.0)
        df["size"] = (params.target_vol / annual_vol.replace(0.0, np.nan)).clip(0.1, 0.95).fillna(0.1)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=80, freq="B")
    close = pd.Series(100.0 + np.arange(80) * 0.1, index=index)
    sample = pd.DataFrame({"open": close, "high": close + 0.2,
                           "low": close - 0.2, "close": close,
                           "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    context = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(sample, strategy.indicators(sample, params), context, params).data
    assert set(result["signal"]).issubset({0, 1}) and result["signal"].iloc[0] == 0
    assert result["size"].gt(0).all()
