from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    frac_order: float = 0.45
    frac_terms: int = 32
    norm_window: int = 40
    corr_window: int = 40
    min_corr: float = 0.15
    entry_z: float = 0.6
    pressure_floor: float = 0.05
    profit_target: float = 0.012
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1790976426"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.frac_terms + params.norm_window + params.corr_window + 3)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (0 < params.frac_order < 1 and params.frac_terms >= 2
                and params.norm_window >= 2 and params.corr_window >= 4
                and 0 <= params.min_corr < 1 and params.entry_z > 0
                and params.pressure_floor > 0 and params.profit_target > 0
                and 3 <= params.hold_bars <= 5):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        weights = [1.0]
        for k in range(1, params.frac_terms):
            weights.append(-weights[-1] * (params.frac_order - k + 1) / k)
        # ponytail: finite fractional memory; increase frac_terms for longer tails.
        out["frac_close"] = sum(
            w * out["log_close"].shift(k) for k, w in enumerate(weights)
        )
        out["frac_mean"] = out["frac_close"].rolling(params.norm_window).mean()
        out["frac_std"] = out["frac_close"].rolling(params.norm_window).std()
        out["frac_z"] = (
            (out["frac_close"] - out["frac_mean"])
            / out["frac_std"].where(out["frac_std"] > 0)
        )
        out["frac_lag1"] = out["frac_z"].shift(1)
        out["rho1"] = out["frac_z"].rolling(params.corr_window).corr(out["frac_lag1"])
        out["primary_forecast"] = out["rho1"] * out["frac_z"]
        out["bar_range"] = data["high"] - data["low"]
        out["pressure"] = (
            (2 * data["close"] - data["high"] - data["low"])
            / out["bar_range"].where(out["bar_range"] > 0)
        )
        out["pressure_lag3"] = out["pressure"].shift(3)
        out["rho3"] = out["pressure"].rolling(params.corr_window).corr(
            out["pressure_lag3"]
        )
        out["pressure_lag2"] = out["pressure"].shift(2)
        out["pressure_forecast"] = out["rho3"] * out["pressure_lag2"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        primary = indicators["primary_forecast"]
        echo = indicators["pressure_forecast"]
        ready = (
            indicators["rho1"].abs().ge(params.min_corr)
            & indicators["rho3"].abs().ge(params.min_corr)
        )
        long_entry = ready & primary.ge(params.entry_z) & echo.ge(params.pressure_floor)
        short_entry = ready & primary.le(-params.entry_z) & echo.le(-params.pressure_floor)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        direction, entry_bar = 0, -1
        entry_price = np.nan
        for i in range(len(data)):
            if direction:
                if np.isnan(entry_price):
                    entry_price, entry_bar = opens[i], i
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = direction * (closes[i] / entry_price - 1) if valid_entry else np.nan
                if (not valid_entry or gain >= params.profit_target
                        or i - entry_bar + 1 >= params.hold_bars):
                    direction, entry_price = 0, np.nan
            elif entries[i]:
                direction, entry_price = int(entries[i]), np.nan
            decisions[i] = direction
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy, params = GeneratedStrategy(), Params()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 97.0,
         "close": 100.0, "volume": 1000.0}, index=index,
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    for side in (1, -1):
        ind = pd.DataFrame(
            {"rho1": 0.5, "rho3": 0.5,
             "primary_forecast": [side, 0, 0, 0, 0, 0],
             "pressure_forecast": [side, 0, 0, 0, 0, 0]}, index=index,
        )
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, side, side, side, side, 0]
        targeted = data.copy()
        targeted.loc[index[1], "close"] = 100 * (1 + side * 0.02)
        result = strategy.generate_signals(targeted, ind, ctx, params).data
        assert result["signal"].tolist() == [0, side, 0, 0, 0, 0]
        ind["pressure_forecast"] *= -1
        assert not strategy.generate_signals(data, ind, ctx, params).data["signal"].any()
    assert not strategy.generate_signals(
        data, strategy.indicators(data, params), ctx, params
    ).data["signal"].any()
