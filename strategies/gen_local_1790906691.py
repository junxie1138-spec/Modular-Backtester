from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ChangeParams:
    ma_window: int = 60
    recent_window: int = 5
    reference_window: int = 20
    change_threshold: float = 1.5
    efficiency_floor: float = 0.35
    profit_target: float = 0.04
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[ChangeParams]):
    strategy_id = "gen_local_1790906691"

    @classmethod
    def params_type(cls):
        return ChangeParams

    def warmup_bars(self, params: ChangeParams) -> int:
        return int(params.ma_window + params.recent_window + params.reference_window + 1)

    def indicators(self, data: pd.DataFrame, params: ChangeParams) -> pd.DataFrame:
        if min(params.ma_window, params.recent_window, params.reference_window) < 2:
            raise ValueError("Windows must be at least two bars.")
        if params.max_hold_bars < 1 or params.profit_target <= 0 or params.change_threshold <= 0:
            raise ValueError("Holding period, target, and change threshold must be positive.")
        if not 0 <= params.efficiency_floor <= 1:
            raise ValueError("Efficiency floor must be between zero and one.")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.shift(1).rolling(params.ma_window).mean()
        out["price_std"] = close.shift(1).rolling(params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["price_std"].replace(0, np.nan)
        recent = out["z"].rolling(params.recent_window)
        reference = out["z"].shift(params.recent_window).rolling(params.reference_window)
        out["recent_z"] = recent.mean()
        out["reference_z"] = reference.mean()
        out["recent_var"] = recent.var(ddof=0)
        out["reference_var"] = reference.var(ddof=0)
        out["pooled_noise"] = np.sqrt((out["recent_var"] + out["reference_var"]) / 2)
        out["change"] = (out["recent_z"] - out["reference_z"]) / out["pooled_noise"].replace(0, np.nan)
        out["displacement"] = close.diff(params.recent_window)
        out["path_length"] = close.diff().abs().rolling(params.recent_window).sum()
        out["efficiency"] = (out["displacement"] / out["path_length"].replace(0, np.nan)).clip(-1, 1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ChangeParams,
    ) -> SignalFrame:
        regime = (
            (indicators["reference_z"] < 0)
            & (indicators["recent_z"] > 0.35)
            & (indicators["change"] >= params.change_threshold)
            & (indicators["efficiency"] >= params.efficiency_floor)
        ).fillna(False)
        entries = (regime & ~regime.shift(1, fill_value=False)).to_numpy()
        strength = (
            (indicators["change"] / (2 * params.change_threshold)).clip(0, 1)
            * indicators["efficiency"].clip(0, 1)
        ).fillna(0)
        conviction = (0.25 + 0.70 * strength).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        entry_bar = -1
        entry_price = np.nan
        held_size = 0.25
        for i in range(len(data)):
            if entry_bar >= 0:
                # The entry open becomes observable only after the decision bar.
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    entry_bar = -1
                elif (
                    closes[i] >= entry_price * (1 + params.profit_target)
                    or i - entry_bar + 1 >= params.max_hold_bars
                ):
                    entry_bar = -1
                else:
                    signals[i] = 1
                    sizes[i] = held_size
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                entry_bar = i + 1
                held_size = float(conviction[i])
                signals[i] = 1
                sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame(
        {"open": 100.0, "high": 106.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    data.loc[index[3], "close"] = 105.0
    derived = pd.DataFrame(
        {"reference_z": 1.0, "recent_z": 1.0, "change": 3.0, "efficiency": 0.5},
        index=index,
    )
    derived.loc[index[[1, 5]], "reference_z"] = -1.0
    derived.loc[index[5], "efficiency"] = 1.0
    strategy = GeneratedStrategy()
    params = ChangeParams(max_hold_bars=3)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, derived, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 1, 1, 1, 0]
    assert 0 < result["size"].iloc[2] < result["size"].iloc[6] <= 0.95
    assert strategy.indicators(data, params).index.equals(data.index)
    assert strategy.generate_signals(data.iloc[:0], derived.iloc[:0], ctx, params).data.empty
    for stop in range(1, len(data) + 1):
        prefix = strategy.generate_signals(data.iloc[:stop], derived.iloc[:stop], ctx, params).data
        pd.testing.assert_frame_equal(prefix, result.iloc[:stop])


if __name__ == "__main__":
    _self_check()
