from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 20
    percentile_window: int = 80
    shock_percentile: float = 0.9
    persistence_bars: int = 3
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790896150"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.ma_window + params.percentile_window + params.persistence_bars + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.ma_window < 2 or params.percentile_window < 2
                or params.persistence_bars < 1 or not 0 < params.shock_percentile < 1
                or not np.isfinite(params.profit_target) or params.profit_target <= 0
                or params.max_hold_bars not in (1, 2)):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        out["ma"] = close.rolling(params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window).std(ddof=0).replace(0, np.nan)
        out["z"] = ((close - out["ma"]) / out["std"]).replace([np.inf, -np.inf], np.nan)
        out["z_change"] = out["z"].diff()
        out["shock_cutoff"] = out["z_change"].abs().rolling(
            params.percentile_window
        ).quantile(params.shock_percentile).shift(1)
        prior = out["z"].shift(1).rolling(params.persistence_bars)
        out["prior_min"] = prior.min()
        out["prior_max"] = prior.max()
        long_entry = ((out["prior_max"] < 0) & (out["z"] > 0)
                      & (out["z_change"] > out["shock_cutoff"]))
        short_entry = ((out["prior_min"] > 0) & (out["z"] < 0)
                       & (-out["z_change"] > out["shock_cutoff"]))
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # The prior close's entry decision fills at this bar's open.
                if age == 0:
                    entry_price = opens[i]
                age += 1
                valid = (np.isfinite(entry_price) and entry_price > 0
                         and np.isfinite(closes[i]) and closes[i] > 0)
                profit = position * (closes[i] / entry_price - 1) if valid else np.nan
                if (not valid or profit >= params.profit_target
                        or age >= params.max_hold_bars):
                    position = 0
                    age = 0
            elif (entries[i] in (-1, 1)
                  and np.isfinite(closes[i]) and closes[i] > 0):
                position = int(entries[i])
                age = 0
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params(profit_target=0.01)
    index = pd.date_range("2020-01-01", periods=7)
    bars = pd.DataFrame({
        "open": 100.0, "high": 103.0, "low": 97.0,
        "close": [100.0, 102.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    forced = pd.DataFrame({"entry": [1, 0, -1, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(bars, forced, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, -1, -1, 0, 0]
    assert (result["size"] > 0).all()
    assert (strategy.indicators(bars, params)["entry"] == 0).all()
