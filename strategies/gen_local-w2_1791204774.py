from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockAcceptanceParams:
    spike_multiple: float = 2.0
    profit_target: float = 0.025


class GeneratedStrategy(BaseStrategy[ShockAcceptanceParams]):
    strategy_id = "gen_local-w2_1791204774"

    @classmethod
    def params_type(cls) -> type[ShockAcceptanceParams]:
        return ShockAcceptanceParams

    def warmup_bars(self, params: ShockAcceptanceParams) -> int:
        return 24

    def indicators(self, data: pd.DataFrame, params: ShockAcceptanceParams) -> pd.DataFrame:
        if not np.isfinite(params.spike_multiple) or params.spike_multiple <= 1.0:
            raise ValueError("spike_multiple must be finite and greater than one")
        if not np.isfinite(params.profit_target) or not 0.0 < params.profit_target < 1.0:
            raise ValueError("profit_target must lie strictly between zero and one")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(20, min_periods=20).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["shock"] = (
            out["prior_atr"].gt(0.0)
            & out["true_range"].gt(params.spike_multiple * out["prior_atr"])
        )
        out["mature_shock"] = out["shock"].shift(2, fill_value=False)
        out["shock_baseline"] = out["atr"].shift(3)
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["shock_midpoint"] = out["midpoint"].shift(2)
        out["previous_range"] = out["true_range"].shift(1)
        out["range_persistence"] = (
            out["true_range"].gt(1.1 * out["shock_baseline"])
            & out["previous_range"].gt(1.1 * out["shock_baseline"])
        )
        out["long_acceptance"] = (
            data["close"].gt(out["shock_midpoint"])
            & out["previous_close"].gt(out["shock_midpoint"])
            & data["close"].gt(out["previous_close"])
        )
        out["short_acceptance"] = (
            data["close"].lt(out["shock_midpoint"])
            & out["previous_close"].lt(out["shock_midpoint"])
            & data["close"].lt(out["previous_close"])
        )
        ready = out["mature_shock"] & out["range_persistence"]
        out["entry"] = np.select([
            ready & out["long_acceptance"],
            ready & out["short_acceptance"],
        ], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockAcceptanceParams,
    ) -> SignalFrame:
        if not np.isfinite(params.profit_target) or not 0.0 < params.profit_target < 1.0:
            raise ValueError("profit_target must lie strictly between zero and one")
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = 0.0
        entry_bar = -1
        pending = False
        cooldown_until = -1
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    entry_price = opens[i]
                    entry_bar = i
                else:
                    position = 0
            if position:
                gain = position * (closes[i] / entry_price - 1.0)
                if (np.isfinite(gain) and gain >= params.profit_target) or i - entry_bar + 1 >= 8:
                    position = 0
                    cooldown_until = i + 3
                decisions[i] = position
                continue
            if i > cooldown_until and entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(entries[i])
                pending = True
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=24)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0, "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[2], "close"] = 103.0
    data.loc[index[22], "close"] = 97.0
    features = pd.DataFrame({"entry": 0}, index=index)
    features.loc[index[0], "entry"] = 1
    features.loc[index[10], "entry"] = -1
    features.loc[index[20], "entry"] = -1
    result = GeneratedStrategy.generate_signals(None, data, features, None, ShockAcceptanceParams()).data
    assert result["signal"].iloc[[0, 1, 3, 11, 18, 19, 21, 23]].tolist() == [0, 1, 0, -1, -1, 0, -1, 0]
    assert result["signal"].isin([-1, 0, 1]).all()
    assert result["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
