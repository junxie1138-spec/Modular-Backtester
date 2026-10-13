from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletRotationParams:
    dominance_ratio: float = 1.75
    profit_target: float = 0.012
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[WaveletRotationParams]):
    strategy_id = "gen_local-w2_1791267630"

    @classmethod
    def params_type(cls):
        return WaveletRotationParams

    @staticmethod
    def warmup_bars(params: WaveletRotationParams) -> int:
        if not np.isfinite(params.dominance_ratio) or params.dominance_ratio <= 1:
            raise ValueError("dominance_ratio must be finite and greater than one")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if isinstance(params.max_hold, bool) or not isinstance(params.max_hold, int) or not 3 <= params.max_hold <= 5:
            raise ValueError("max_hold must be an integer from three to five")
        return 10

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveletRotationParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        out["gap_return"] = (opening / closing.shift(1) - 1).replace([np.inf, -np.inf], np.nan)
        out["session_return"] = (closing / opening - 1).replace([np.inf, -np.inf], np.nan)
        for name in ("gap", "session"):
            r = out[name + "_return"]
            recent = r.rolling(4, min_periods=4).sum()
            out[name + "_recent"] = recent
            out[name + "_haar2"] = (r - r.shift(1)) / np.sqrt(2.0)
            pair = r.rolling(2, min_periods=2).sum()
            out[name + "_haar4"] = (pair - pair.shift(2)) / 2.0
            out[name + "_haar8"] = (recent - recent.shift(4)) / np.sqrt(8.0)
            out[name + "_energy"] = (
                out[name + "_haar2"].pow(2)
                + out[name + "_haar4"].pow(2)
                + out[name + "_haar8"].pow(2)
            )
        out["dominance"] = out["gap_energy"] / out["session_energy"].where(out["session_energy"] > 0)
        out["dominance"] = out["dominance"].replace([np.inf, -np.inf], np.nan)
        out["previous_dominance"] = out["dominance"].shift(1)
        change = (
            (out["dominance"] >= params.dominance_ratio)
            & (out["previous_dominance"] < params.dominance_ratio)
            & (out["gap_haar4"] * out["gap_haar8"] > 0)
            & (out["gap_recent"] * out["gap_haar8"] > 0)
            & (out["session_haar8"] * out["gap_haar8"] < 0)
        )
        out["entry_direction"] = np.sign(out["gap_recent"]).where(change, 0).fillna(0).astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletRotationParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        entries = indicators["entry_direction"].fillna(0).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position = 0
        entry_price = np.nan
        fill_bar = -1
        for i in range(len(data)):
            if position:
                # Entry ordered at the preceding close fills at this open.
                if i == fill_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                if position:
                    gain = position * (closes[i] / entry_price - 1.0)
                    if gain >= params.profit_target or i - fill_bar + 1 >= params.max_hold:
                        # Close-based exit decision fills at the next open.
                        position = 0
            elif entries[i] in (-1, 1):
                position = int(entries[i])
                entry_price = np.nan
                fill_bar = i + 1
            raw[i] = position
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = WaveletRotationParams(max_hold=3, profit_target=0.01)
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 97.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    entries = pd.DataFrame({"entry_direction": [1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0]}, index=index)
    timed = GeneratedStrategy.generate_signals(data, entries, None, params).data
    assert timed["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0]
    data.loc[index[2], "close"] = 102.0
    data.loc[index[8], "close"] = 98.0
    targeted = GeneratedStrategy.generate_signals(data, entries, None, params).data
    assert targeted["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, -1, -1, 0, 0, 0]
    derived = GeneratedStrategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived["entry_direction"].iloc[:9].any()
    assert GeneratedStrategy.warmup_bars(params) == 10
    assert (targeted["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
