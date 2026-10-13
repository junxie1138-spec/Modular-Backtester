from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791293988"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 24

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        if isinstance(params.max_hold, bool) or not isinstance(params.max_hold, int) or params.max_hold < 1:
            raise ValueError("max_hold must be a positive integer")

        out = pd.DataFrame(index=data.index)
        high = data["high"].astype(float).replace([np.inf, -np.inf], np.nan)
        low = data["low"].astype(float).replace([np.inf, -np.inf], np.nan)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        valid = high.ge(low) & low.gt(0) & close.between(low, high)
        high = high.where(valid)
        low = low.where(valid)
        close = close.where(valid)

        out["midpoint"] = (high + low) / 2.0
        out["daily_range"] = high - low
        out["range_fast"] = out["daily_range"].rolling(5, min_periods=5).mean()
        out["range_slow"] = out["daily_range"].rolling(20, min_periods=20).mean()
        out["compression"] = out["range_fast"] / out["range_slow"].replace(0.0, np.nan)

        # Twelve-bar least-squares midpoint slope, with time centered at 5.5.
        out["latent_slope"] = sum(
            (j - 5.5) * out["midpoint"].shift(11 - j)
            for j in range(12)
        ) / 143.0
        out["expected_slope"] = out["latent_slope"].shift(1).rolling(3, min_periods=3).mean()
        out["slope_innovation"] = out["latent_slope"] - out["expected_slope"]
        out["innovation_sum"] = out["slope_innovation"].rolling(3, min_periods=3).sum()
        out["innovation_pressure"] = out["innovation_sum"] / out["range_slow"].replace(0.0, np.nan)
        out["prior_high"] = high.shift(1).rolling(20, min_periods=20).max()
        out["prior_low"] = low.shift(1).rolling(20, min_periods=20).min()
        out = out.replace([np.inf, -np.inf], np.nan)

        out["setup"] = (
            valid
            & out["compression"].le(0.70)
            & out["latent_slope"].gt(0.0)
            & out["slope_innovation"].gt(0.0)
            & out["innovation_pressure"].gt(0.12)
            & close.lt(out["prior_high"])
            & close.gt(out["prior_low"])
        ).fillna(False)
        out["entry"] = out["setup"] & ~out["setup"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        if isinstance(params.max_hold, bool) or not isinstance(params.max_hold, int) or params.max_hold < 1:
            raise ValueError("max_hold must be a positive integer")

        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if active:
                # This open is available only after the prior close's entry decision.
                if i == entry_bar:
                    entry_price = opening[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                target_hit = (
                    valid_entry
                    and np.isfinite(close[i])
                    and close[i] >= entry_price * (1.0 + params.profit_target)
                )
                expired = i - entry_bar + 1 >= params.max_hold
                if not valid_entry or target_hit or expired:
                    active = False
                decisions[i] = int(active)
            elif entry[i] and np.isfinite(close[i]) and close[i] > 0.0:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                decisions[i] = 1

        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 99.0,
         "close": [100.0, 100.0, 103.0, 100.0, 100.0, 100.0], "volume": 1000.0},
        index=index,
    )
    features = pd.DataFrame({"entry": [True, False, False, False, False, False]}, index=index)
    target = GeneratedStrategy.generate_signals(None, data, features, None, Params())
    assert target.data["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    data["close"] = 100.0
    timed = GeneratedStrategy.generate_signals(None, data, features, None, Params(max_hold=2))
    assert timed.data["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert timed.data["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
