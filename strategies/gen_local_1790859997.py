from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    min_streak: int = 2
    profit_target: float = 0.006


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790859997"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.min_streak) + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.min_streak < 1 or int(params.min_streak) != params.min_streak:
            raise ValueError("min_streak must be a positive integer")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & data["close"].gt(0))
        volume = data["volume"].where(np.isfinite(data["volume"]) & data["volume"].gt(0))
        out["close_change"] = close.diff()
        out["volume_change"] = volume.diff()
        out["sell_pressure"] = out["close_change"].lt(0) & out["volume_change"].gt(0)
        pressure = out["sell_pressure"].astype(int)
        out["pressure_streak"] = pressure.groupby((~out["sell_pressure"]).cumsum()).cumsum().clip(upper=params.min_streak)
        out["entry_setup"] = (
            out["pressure_streak"].shift(1).ge(params.min_streak)
            & out["close_change"].lt(0)
            & out["close_change"].abs().lt(out["close_change"].shift(1).abs())
            & out["volume_change"].lt(0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        setup = indicators["entry_setup"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            # A previous close's entry decision fills at this session's open.
            if entry_bar < 0 and i > 0 and decisions[i - 1] == 1:
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_bar = i
                    entry_price = opens[i] * 1.0005  # Configured five-basis-point entry slippage.
                else:
                    continue
            if entry_bar >= 0:
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                # Exit decisions fill next open: two completed sessions at most.
                if target_hit or i - entry_bar + 1 >= 2:
                    entry_bar = -1
                else:
                    decisions[i] = 1
            elif setup[i] and i >= self.warmup_bars(params) - 1:
                decisions[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7, freq="B")
    data = pd.DataFrame({
        "open": [100, 100, 99, 98, 97.5, 97.5, 97.5],
        "high": [101, 101, 100, 99, 99, 99, 99],
        "low": [99, 98, 97, 97, 97, 97, 97],
        "close": [100, 99, 98, 97.5, 98.5, 98.6, 98.4],
        "volume": [100, 120, 140, 100, 90, 90, 90],
    }, index=index)
    strategy = GeneratedStrategy()
    params = Params()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 1, 0, 0]
    data.loc[index[4:], "close"] = [97.6, 97.7, 97.8]
    result = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 1, 1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
