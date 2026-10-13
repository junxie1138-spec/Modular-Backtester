from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    shock_share: float = 0.65
    min_drawdown: float = 0.015
    profit_target: float = 0.02
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791312299"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        x = np.log(close.where(close > 0))
        fine = (x - x.shift(1)) / np.sqrt(2.0)
        medium = (x + x.shift(1) - x.shift(2) - x.shift(3)) / 2.0
        coarse = (
            x + x.shift(1) + x.shift(2) + x.shift(3)
            - x.shift(4) - x.shift(5) - x.shift(6) - x.shift(7)
        ) / np.sqrt(8.0)
        fine_energy = (
            fine.pow(2) + fine.shift(2).pow(2)
            + fine.shift(4).pow(2) + fine.shift(6).pow(2)
        )
        medium_energy = medium.pow(2) + medium.shift(4).pow(2)
        coarse_energy = coarse.pow(2)
        total_energy = fine_energy + medium_energy + coarse_energy
        localization = fine.clip(upper=0).pow(2) / fine_energy.where(fine_energy > 0)
        fine_share = fine_energy / total_energy.where(total_energy > 0)
        peak = data["high"].rolling(8, min_periods=8).max()
        drawdown = close / peak.where(peak > 0) - 1.0
        previous_midpoint = (data["high"].shift(1) + data["low"].shift(1)) / 2.0
        entry = (
            (localization.shift(1) >= params.shock_share)
            & (fine_share.shift(1) >= 0.45)
            & (coarse.shift(1) < 0)
            & (drawdown.shift(1) <= -params.min_drawdown)
            & (close > previous_midpoint)
            & (close > close.shift(1))
            & (close > data["open"])
        )
        out = pd.DataFrame({
            "log_close": x,
            "fine": fine,
            "medium": medium,
            "coarse": coarse,
            "fine_energy": fine_energy,
            "medium_energy": medium_energy,
            "coarse_energy": coarse_energy,
            "total_energy": total_energy,
            "localization": localization,
            "fine_share": fine_share,
            "peak": peak,
            "drawdown": drawdown,
            "previous_midpoint": previous_midpoint,
            "entry": entry.fillna(False).astype(int),
        }, index=data.index)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        if not (0 < params.shock_share <= 1 and 0 < params.min_drawdown < 1):
            raise ValueError("Invalid entry parameters")
        if not (params.profit_target > 0 and params.max_hold_bars >= 1):
            raise ValueError("Invalid exit parameters")
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        active = False
                        continue
                elapsed = i - entry_bar + 1
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if target_hit or elapsed >= params.max_hold_bars:
                    active = False
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({
        "open": 100.0, "high": 104.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    entries = pd.DataFrame({"entry": [1] + [0] * 9}, index=index)
    data.iloc[2, data.columns.get_loc("close")] = 103.0
    result = strategy.generate_signals(data, entries, None, RecoveryParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(
        data, entries, None, RecoveryParams(max_hold_bars=2)
    ).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert (result["size"] > 0).all()
    computed = strategy.indicators(data, RecoveryParams())
    assert computed.index.equals(data.index)
    assert np.isfinite(computed.to_numpy(dtype=float)).all()
    assert not computed["entry"].any()
    assert strategy.warmup_bars(RecoveryParams()) == 9


if __name__ == "__main__":
    _self_check()
