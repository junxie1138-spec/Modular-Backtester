from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletSeasonParams:
    coarse_share: float = 0.60
    profit_target: float = 0.008
    hold_bars: int = 4

    def __post_init__(self):
        if not 0.0 < self.coarse_share < 1.0:
            raise ValueError("coarse_share must lie between zero and one")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if self.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be 3, 4, or 5")


class GeneratedStrategy(BaseStrategy[WaveletSeasonParams]):
    strategy_id = "gen_local-w2_1790966158"

    @classmethod
    def params_type(cls):
        return WaveletSeasonParams

    def warmup_bars(self, params: WaveletSeasonParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: WaveletSeasonParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = np.log(close).diff()
        r = out["return"]
        out["haar_2"] = (r - r.shift(1)) / np.sqrt(2.0)
        out["haar_4"] = (r.rolling(2).sum() - r.rolling(2).sum().shift(2)) / 2.0
        out["haar_8"] = (r.rolling(4).sum() - r.rolling(4).sum().shift(4)) / np.sqrt(8.0)
        for scale in (2, 4, 8):
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2)
        total = out[["energy_2", "energy_4", "energy_8"]].sum(axis=1, min_count=3)
        out["coarse_share"] = out["energy_8"] / total.where(total > 0)
        out["ready"] = r.rolling(8).count().eq(8) & total.notna()
        out["early_week"] = data.index.dayofweek <= 1
        out["entry"] = (
            out["ready"]
            & out["early_week"]
            & out["haar_8"].lt(0)
            & out["return"].gt(0)
            & out["coarse_share"].ge(params.coarse_share)
        )
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletSeasonParams,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=bool)
        active = False
        entry_price = 0.0
        entry_bar = -1
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Yesterday's decision fills at today's open.
                    entry_price = opens[i]
                valid = np.isfinite(entry_price) and entry_price > 0
                target_hit = valid and np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                expired = i - entry_bar + 1 >= params.hold_bars
                if not valid or target_hit or expired:
                    active = False
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = i + 1
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = WaveletSeasonParams()
    index = pd.bdate_range("2024-01-01", periods=16)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = strategy.indicators(data, params)
    assert not indicators.isna().any().any()
    assert not indicators["entry"].any()
    assert strategy.warmup_bars(params) == 9
    indicators["entry"] = False
    indicators.iloc[9, indicators.columns.get_loc("entry")] = True
    timed = strategy.generate_signals(data, indicators, None, params).data["signal"]
    assert timed.tolist() == [0] * 10 + [1] * 4 + [0] * 2
    data.iloc[11, data.columns.get_loc("close")] = 101.0
    targeted = strategy.generate_signals(data, indicators, None, params).data["signal"]
    assert targeted.tolist() == [0] * 10 + [1] * 2 + [0] * 4


if __name__ == "__main__":
    _self_check()
