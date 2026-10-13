from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    fine_drop: float = 0.15
    middle_share: float = 0.30
    profit_target: float = 0.04
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791239717"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return 10

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if not (0 < params.fine_drop < 1 and 0 < params.middle_share < 1):
            raise ValueError("Energy thresholds must lie between zero and one.")
        out = pd.DataFrame(index=data.index)
        out["return"] = np.log(data["close"].where(data["close"] > 0)).diff()
        r = out["return"]
        out["haar2"] = (r - r.shift(1)) / np.sqrt(2.0)
        out["haar4"] = (r + r.shift(1) - r.shift(2) - r.shift(3)) / 2.0
        out["haar8"] = (r.rolling(4).mean() - r.shift(4).rolling(4).mean()) * np.sqrt(2.0)
        out["fine_energy"] = sum(out["haar2"].shift(k).pow(2) for k in (0, 2, 4, 6))
        out["middle_energy"] = out["haar4"].pow(2) + out["haar4"].shift(4).pow(2)
        out["coarse_energy"] = out["haar8"].pow(2)
        out["total_energy"] = out["fine_energy"] + out["middle_energy"] + out["coarse_energy"]
        total = out["total_energy"].where(out["total_energy"] > 0)
        out["fine_share"] = out["fine_energy"] / total
        out["middle_share"] = out["middle_energy"] / total
        out["previous_fine_share"] = out["fine_share"].shift(1)
        out["previous_middle_energy"] = out["middle_energy"].shift(1)
        out["prior_high"] = data["high"].shift(1).rolling(7).max()
        out["ready"] = out[["fine_share", "middle_share", "previous_fine_share",
                            "previous_middle_energy", "prior_high"]].notna().all(axis=1)
        out["entry"] = (
            out["ready"]
            & (data["close"] > out["prior_high"])
            & (out["previous_fine_share"] - out["fine_share"] >= params.fine_drop)
            & (out["middle_share"] >= params.middle_share)
            & (out["middle_energy"] > out["previous_middle_energy"])
        )
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: WaveletParams,
    ) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite.")
        if not isinstance(params.hold_bars, int) or params.hold_bars < 1:
            raise ValueError("hold_bars must be a positive integer.")
        entry = indicators["entry"].to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = 0.0
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        entry_bar = -1
                        continue
                # Exit decisions at the close execute at the following open.
                hit_target = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if hit_target or i - entry_bar + 1 >= params.hold_bars:
                    entry_bar = -1
                else:
                    signal[i] = 1
            elif entry[i] and np.isfinite(closes[i]) and closes[i] > 0:
                entry_bar = i + 1
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    bars = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=12),
    )
    entries = pd.DataFrame({"entry": [True] + [False] * 11}, index=bars.index)
    context = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=10)
    params = WaveletParams(hold_bars=3)
    result = strategy.generate_signals(bars, entries, context, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1] + [0] * 8
    bars.loc[bars.index[2], "close"] = 105.0
    result = strategy.generate_signals(bars, entries, context, params).data
    assert result["signal"].tolist() == [0, 1, 1] + [0] * 9
    derived = strategy.indicators(bars, params)
    assert derived.index.equals(bars.index) and not derived.isna().any().any()
    assert not derived["entry"].iloc[:9].any()
