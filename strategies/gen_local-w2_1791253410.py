from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791253410"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return 25

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        r = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        interior_min = r.shift(1)
        degree = pd.Series(0.0, index=data.index)
        cross = pd.Series(0.0, index=data.index)
        energy = pd.Series(0.0, index=data.index)
        # Horizontal visibility on inverted returns: endpoints lie below every intervening return.
        # Exclude adjacent edges so topology selects nonlocal return troughs.
        for lag in range(2, 24):
            past = r.shift(lag)
            successor = r.shift(lag - 1)
            edge = (r < interior_min) & (past < interior_min)
            degree += edge.astype(float)
            cross += (past * successor).where(edge, 0.0).fillna(0.0)
            energy += past.pow(2).where(edge, 0.0).fillna(0.0)
            interior_min = pd.Series(
                np.minimum(interior_min.to_numpy(), past.to_numpy()), index=data.index
            )
        ready = r.rolling(24, min_periods=24).count().eq(24)
        out["return"] = r
        out["downside_degree"] = degree.where(ready)
        out["transition_cross"] = cross.where(ready)
        out["transition_energy"] = energy.where(ready)
        # Graph-conditioned lag-one regression; negative coefficients forecast reversal.
        out["transition_coefficient"] = (
            out["transition_cross"] / out["transition_energy"].replace(0.0, np.nan)
        ).clip(-1.0, 1.0)
        out["rebound_forecast"] = r * out["transition_coefficient"]
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError("profit_target must be finite and positive")
        if not isinstance(params.max_hold, int) or isinstance(params.max_hold, bool) or params.max_hold < 1:
            raise ValueError("max_hold must be a positive integer")
        entry = (
            indicators["return"].lt(0.0)
            & indicators["downside_degree"].ge(3.0)
            & indicators["rebound_forecast"].gt(0.0015)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        fill_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                if i == fill_bar:
                    # Read the fill open only after that bar has arrived.
                    entry_price = opens[i]
                invalid_fill = not np.isfinite(entry_price) or entry_price <= 0.0
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                timed_out = i - fill_bar + 1 >= params.max_hold
                if invalid_fill or target_hit or timed_out:
                    active = False
                else:
                    raw[i] = 1
            elif entry[i]:
                active = True
                fill_bar = i + 1
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    features = pd.DataFrame(
        {"return": 0.0, "downside_degree": 0.0, "rebound_forecast": 0.0}, index=index
    )
    features.iloc[0] = [-0.01, 3.0, 0.002]
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=25)
    timed = strategy.generate_signals(data, features, ctx, GeneratedParams(max_hold=2)).data
    assert timed["signal"].tolist() == [0, 1, 1] + [0] * 9
    data.loc[index[3], "close"] = 103.0
    profit = strategy.generate_signals(data, features, ctx, GeneratedParams()).data
    assert profit["signal"].tolist() == [0, 1, 1, 1] + [0] * 8


if __name__ == "__main__":
    _self_check()
