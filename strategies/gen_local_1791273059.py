from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    channel_bars: int = 20
    area_bars: int = 5
    area_threshold: float = 0.04
    position_ceiling: float = 0.55
    settlement_floor: float = 0.60
    profit_target: float = 0.006
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791273059"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(params.channel_bars + params.area_bars + 1)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if params.channel_bars < 2 or params.area_bars < 2:
            raise ValueError("Channel and area windows must be at least two bars")
        if params.max_hold_bars not in (1, 2) or params.profit_target <= 0:
            raise ValueError("Require a positive target and a one- or two-bar holding limit")
        if not (0 < params.position_ceiling < 1 and 0 < params.settlement_floor < 1):
            raise ValueError("Position and settlement thresholds must lie inside (0, 1)")
        if not np.isfinite(params.area_threshold) or params.area_threshold <= 0:
            raise ValueError("Area threshold must be positive and finite")
        out = pd.DataFrame(index=data.index)
        out["channel_low"] = data["low"].rolling(params.channel_bars).min()
        out["channel_high"] = data["high"].rolling(params.channel_bars).max()
        out["channel_width"] = (out["channel_high"] - out["channel_low"]).where(lambda s: s > 0)
        out["daily_range"] = (data["high"] - data["low"]).where(lambda s: s > 0)
        out["position"] = (data["close"] - out["channel_low"]) / out["channel_width"]
        out["settlement"] = (data["close"] - data["low"]) / out["daily_range"]
        x = out["position"]
        y = out["settlement"]
        out["position_change"] = x.diff()
        out["area_edge"] = x.shift(1) * y - y.shift(1) * x
        # Closing the path makes its signed area translation invariant.
        out["signature_area"] = 0.5 * (
            out["area_edge"].rolling(params.area_bars).sum()
            - x.shift(params.area_bars) * y
            + y.shift(params.area_bars) * x
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        out["setup"] = (
            out["signature_area"].ge(params.area_threshold)
            & out["position"].lt(params.position_ceiling)
            & out["settlement"].ge(params.settlement_floor)
            & out["position_change"].gt(0)
        ).fillna(False)
        out["confirmed"] = out["setup"] & out["setup"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool).copy()
        confirmed[:self.warmup_bars(params) - 1] = False
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    entry_price = opens[i]
                # Observe the target at the close; the shifted exit fills next bar.
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                time_up = i - entry_bar + 1 >= params.max_hold_bars
                if not np.isfinite(entry_price) or entry_price <= 0 or target_hit or time_up:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    signal[i] = 1
            elif confirmed[i] and np.isfinite(closes[i]) and closes[i] > 0:
                entry_bar = i + 1
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RecoveryParams(channel_bars=3, area_bars=2)
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    setup = pd.Series(False, index=index)
    setup.iloc[4:6] = True
    indicators = pd.DataFrame({"confirmed": setup & setup.shift(1, fill_value=False)})
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    data.loc[index[6], "close"] = 101.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    assert result["size"].gt(0).all()
    x = pd.Series([0.0, 1.0, 1.0])
    y = pd.Series([0.0, 0.0, 1.0])
    area = 0.5 * ((x.shift(1) * y - y.shift(1) * x).rolling(2).sum()
                  - x.shift(2) * y + y.shift(2) * x)
    assert area.iloc[-1] == 0.5


if __name__ == "__main__":
    _self_check()
