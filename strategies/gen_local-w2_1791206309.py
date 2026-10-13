from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_bars: int = 20
    entry_z: float = 1.25
    persistence_bars: int = 3
    profit_target: float = 0.02
    max_hold_bars: int = 8

    def __post_init__(self):
        if self.ma_bars < 2 or self.persistence_bars < 1:
            raise ValueError("Lookbacks must be positive; ma_bars must exceed one.")
        if not np.isfinite(self.entry_z) or self.entry_z <= 0:
            raise ValueError("entry_z must be positive and finite.")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite.")
        if self.max_hold_bars < 1:
            raise ValueError("max_hold_bars must be positive.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791206309"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.ma_bars + params.persistence_bars - 1, 3))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        volume = data["volume"]
        out["ma"] = close.rolling(params.ma_bars, min_periods=params.ma_bars).mean()
        out["std"] = close.rolling(params.ma_bars, min_periods=params.ma_bars).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["regime_ma"] = close.rolling(200, min_periods=200).mean()
        out["oversold_count"] = out["z"].le(-params.entry_z).astype(int).rolling(params.persistence_bars, min_periods=params.persistence_bars).sum()
        out["volume_fades"] = volume.gt(0) & volume.lt(volume.shift(1)) & volume.shift(1).lt(volume.shift(2))
        out["entry"] = (close.gt(out["regime_ma"]) & out["oversold_count"].eq(params.persistence_bars) & out["volume_fades"] & out["z"].notna()).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # The pending order fills at this bar's open, never a future open.
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        entry_bar = -1
                        continue
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or expired:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                signals[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": [100.0, 110.0, 110.0, 110.0, 110.0], "close": [100.0, 100.0, 103.0, 113.0, 113.0]}, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, Params())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0]
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, Params(profit_target=0.5, max_hold_bars=2))
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
