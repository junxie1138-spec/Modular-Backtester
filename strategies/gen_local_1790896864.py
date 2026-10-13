from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    fine_share_min: float = 0.50
    shock_share_min: float = 0.60
    profit_target: float = 0.006
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1790896864"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return 8

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        x = out["log_close"].replace([np.inf, -np.inf], np.nan)
        # Complete orthonormal Haar detail energies of the trailing eight closes.
        out["detail_2"] = (x - x.shift(1)) / np.sqrt(2.0)
        out["detail_4"] = (x + x.shift(1) - x.shift(2) - x.shift(3)) / 2.0
        out["detail_8"] = (
            x.rolling(4, min_periods=4).sum()
            - x.shift(4).rolling(4, min_periods=4).sum()
        ) / np.sqrt(8.0)
        d2 = out["detail_2"].pow(2)
        d4 = out["detail_4"].pow(2)
        out["fine_energy"] = d2 + d2.shift(2) + d2.shift(4) + d2.shift(6)
        out["medium_energy"] = d4 + d4.shift(4)
        out["coarse_energy"] = out["detail_8"].pow(2)
        out["total_energy"] = (
            out["fine_energy"] + out["medium_energy"] + out["coarse_energy"]
        )
        out["fine_share"] = (
            out["fine_energy"] / out["total_energy"].replace(0.0, np.nan)
        ).fillna(0.0)
        out["shock_share"] = (
            d2 / out["fine_energy"].replace(0.0, np.nan)
        ).fillna(0.0)
        eligible = (
            (out["fine_share"] >= params.fine_share_min)
            & (out["shock_share"] >= params.shock_share_min)
            & (out["detail_2"] * out["detail_4"] > 0.0)
            & out["total_energy"].notna()
        )
        out["entry"] = np.where(
            eligible, -np.sign(out["detail_2"].fillna(0.0)), 0
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        if not (0.0 < params.fine_share_min <= 1.0
                and 0.0 < params.shock_share_min <= 1.0
                and np.isfinite(params.profit_target)
                and params.profit_target > 0.0
                and params.max_hold_bars in (1, 2)):
            raise ValueError("Invalid energy thresholds, target, or holding horizon")
        decisions = np.zeros(len(data), dtype=int)
        direction, held = 0, 0
        entry_price = np.nan
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        for i in range(len(data)):
            if direction:
                # The prior close's entry decision fills at this session's open.
                if held == 0:
                    entry_price = opens[i]
                held += 1
                valid = (
                    np.isfinite(entry_price) and entry_price > 0.0
                    and np.isfinite(closes[i]) and closes[i] > 0.0
                )
                gain = direction * (closes[i] / entry_price - 1.0) if valid else 0.0
                if (not valid or gain >= params.profit_target
                        or held >= params.max_hold_bars):
                    direction, held = 0, 0
                    entry_price = np.nan
                    # Keep this decision flat; no same-close re-entry.
            elif np.isfinite(closes[i]) and closes[i] > 0.0:
                direction = int(entries[i])
                held = 0
            decisions[i] = direction
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 97.0,
         "close": [100.0, 102.0, 100.0, 100.0, 100.0, 100.0, 100.0],
         "volume": 1000.0}, index=index
    )
    strategy, params = GeneratedStrategy(), WaveletParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=8)
    entries = pd.DataFrame({"entry": [1, 0, 0, -1, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, entries, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, -1, 0]
    entries["entry"] = [-1, 0, 0, 0, 0, 0, 0]
    data.loc[index[1], "close"] = 98.0
    result = strategy.generate_signals(data, entries, ctx, params).data
    assert result["signal"].tolist() == [0, -1, 0, 0, 0, 0, 0]
    assert strategy.indicators(data, params)["entry"].eq(0).all()
