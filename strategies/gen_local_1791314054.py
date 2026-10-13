from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PersistenceParams:
    transition_window: int = 24
    percentile_window: int = 126
    entry_percentile: float = 0.85
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[PersistenceParams]):
    strategy_id = "gen_local_1791314054"

    @classmethod
    def params_type(cls):
        return PersistenceParams

    def warmup_bars(self, params: PersistenceParams) -> int:
        return int(params.transition_window + params.percentile_window + 2)

    def indicators(self, data: pd.DataFrame, params: PersistenceParams) -> pd.DataFrame:
        if params.transition_window < 2 or params.percentile_window < 2:
            raise ValueError("Rolling windows must be at least two bars.")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must lie strictly between zero and one.")
        if params.profit_target <= 0.0 or params.max_hold_bars < 1:
            raise ValueError("Profit target and holding limit must be positive.")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        out["previous_return"] = out["return"].shift(1)
        valid = out["return"].notna() & out["previous_return"].notna()
        transitions = {
            "up_trials": out["previous_return"].gt(0),
            "down_trials": out["previous_return"].lt(0),
            "up_repeats": out["previous_return"].gt(0) & out["return"].gt(0),
            "down_repeats": out["previous_return"].lt(0) & out["return"].lt(0),
        }
        for name, event in transitions.items():
            out[name] = event.astype(float).where(valid).rolling(
                params.transition_window, min_periods=params.transition_window
            ).sum()
        out["up_persistence"] = (out["up_repeats"] + 1.0) / (out["up_trials"] + 2.0)
        out["down_persistence"] = (out["down_repeats"] + 1.0) / (out["down_trials"] + 2.0)
        out["strength"] = out["up_persistence"] - out["down_persistence"]
        out["absolute_strength"] = out["strength"].abs()
        out["threshold"] = out["absolute_strength"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.entry_percentile)
        qualified = (
            out["absolute_strength"].gt(out["threshold"])
            & out["strength"].mul(out["return"]).gt(0)
            & out["threshold"].notna()
        )
        out["entry_direction"] = np.where(
            qualified, np.sign(out["strength"].fillna(0)), 0
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PersistenceParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        candidates = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position:
                # Yesterday's entry decision fills at today's open.
                if entry_bar < 0:
                    entry_price = opens[i]
                    entry_bar = i
                age = i - entry_bar + 1
                gain = position * (closes[i] / entry_price - 1.0)
                if (
                    not np.isfinite(entry_price)
                    or entry_price <= 0.0
                    or not np.isfinite(closes[i])
                    or gain >= params.profit_target
                    or age >= params.max_hold_bars
                ):
                    position = 0
                    entry_bar = -1
                    # Keep the exit decision flat even if another entry qualifies.
                    continue
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(candidates[i])
                entry_bar = -1
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = PersistenceParams()
    index = pd.date_range("2020-01-01", periods=12)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction in (1, -1):
        for take_profit in (False, True):
            data = pd.DataFrame(
                {"open": 100.0, "high": 104.0, "low": 96.0,
                 "close": 100.0, "volume": 1000.0}, index=index
            )
            if take_profit:
                data.loc[index[3], "close"] = 100.0 * (1.0 + direction * 0.03)
            indicators = pd.DataFrame({"entry_direction": direction}, index=index)
            signals = strategy.generate_signals(data, indicators, ctx, params).data
            exit_bar = 4 if take_profit else params.max_hold_bars + 1
            assert signals["signal"].iloc[0] == 0
            assert (signals["signal"].iloc[1:exit_bar] == direction).all()
            assert signals["signal"].iloc[exit_bar] == 0
            assert signals["size"].gt(0).all()
    flat = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0,
         "close": 100.0, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=200),
    )
    assert strategy.indicators(flat, params)["entry_direction"].eq(0).all()


if __name__ == "__main__":
    _self_check()
