from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationParams:
    vol_window: int = 20
    accel_floor: float = 0.15
    target_vol: float = 0.012
    profit_target: float = 0.006
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[AccelerationParams]):
    strategy_id = "gen_local-w2_1791300093"

    @classmethod
    def params_type(cls):
        return AccelerationParams

    def warmup_bars(self, params: AccelerationParams) -> int:
        return params.vol_window + 1

    def indicators(self, data: pd.DataFrame, params: AccelerationParams) -> pd.DataFrame:
        if params.vol_window < 2 or params.accel_floor < 0 or params.target_vol <= 0 or params.profit_target <= 0 or params.max_hold < 1:
            raise ValueError("Invalid strategy parameters")
        ret = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        vol = ret.rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0)
        score = (ret.diff() / vol.where(vol > 0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        recovery = data["open"].lt(data["close"].shift(1)) & data["close"].gt(data["open"])
        qualifying = score.gt(params.accel_floor) & recovery
        out = pd.DataFrame(index=data.index)
        out["acceleration"] = score
        out["volatility"] = vol.fillna(0.0)
        out["setup"] = (qualifying & qualifying.shift(1, fill_value=False)).fillna(False)
        out["size"] = (params.target_vol / vol.where(vol > 0)).clip(0.25, 1.0).fillna(0.25).astype(float)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AccelerationParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        setup = indicators["setup"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        held = 0
        entry_open = 0.0
        for i in range(self.warmup_bars(params), len(data)):
            if active:
                if held == 0:
                    entry_open = opens[i]
                held += 1
                if closes[i] >= entry_open * (1.0 + params.profit_target) or held >= params.max_hold:
                    active = False
            elif setup[i]:
                active = True
                held = 0
            raw[i] = int(active)
        df = pd.DataFrame({"signal": raw, "size": indicators["size"]}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=27, freq="B")
    bars = pd.DataFrame({"open": 100.0, "close": 100.0}, index=index)
    bars.iloc[23, bars.columns.get_loc("close")] = 101.0
    test_indicators = pd.DataFrame({"setup": False, "size": 0.5}, index=index)
    test_indicators.iloc[22, test_indicators.columns.get_loc("setup")] = True
    strategy = GeneratedStrategy()
    profit = strategy.generate_signals(bars, test_indicators, None, AccelerationParams()).data
    timed = strategy.generate_signals(bars, test_indicators, None, AccelerationParams(profit_target=0.99)).data
    assert profit["signal"].iloc[23:25].tolist() == [1, 0]
    assert timed["signal"].iloc[23:26].tolist() == [1, 1, 0]
    assert profit["size"].gt(0).all()
