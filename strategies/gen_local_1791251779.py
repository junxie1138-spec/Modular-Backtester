from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MonthEdgeParams:
    volume_window: int = 20
    history_window: int = 252
    seasonal_min_obs: int = 20
    volume_percentile: float = 0.75
    profit_target: float = 0.04
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[MonthEdgeParams]):
    strategy_id = "gen_local_1791251779"

    @classmethod
    def params_type(cls) -> type[MonthEdgeParams]:
        return MonthEdgeParams

    def warmup_bars(self, params: MonthEdgeParams) -> int:
        return int(params.volume_window + params.history_window + 1)

    def indicators(self, data: pd.DataFrame, params: MonthEdgeParams) -> pd.DataFrame:
        if not (
            params.volume_window >= 1
            and params.history_window >= params.seasonal_min_obs >= 1
            and 0 < params.volume_percentile < 1
            and params.profit_target > 0
            and params.holding_bars >= 1
        ):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        volume = data["volume"].where(data["volume"] >= 0)
        out["return"] = close.pct_change(fill_method=None)
        out["intraday_return"] = close / opening - 1.0
        out["month_edge"] = (data.index.day <= 5) | (data.index.day >= 25)
        out["volume_baseline"] = volume.shift(1).rolling(params.volume_window).mean()
        out["relative_volume"] = volume / out["volume_baseline"].where(out["volume_baseline"] > 0)
        out["seasonal_volume"] = out["relative_volume"].where(out["month_edge"])
        out["volume_threshold"] = out["seasonal_volume"].shift(1).rolling(
            params.history_window, min_periods=params.seasonal_min_obs
        ).quantile(params.volume_percentile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: MonthEdgeParams,
    ) -> SignalFrame:
        eligible = (
            indicators["month_edge"].fillna(False).astype(bool)
            & (indicators["relative_volume"] > indicators["volume_threshold"])
            & (indicators["return"] * indicators["intraday_return"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        direction = np.sign(indicators["return"].fillna(0)).to_numpy()
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        position, held = 0, 0
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # Read the entry open only when its bar has arrived.
                if np.isnan(entry_price):
                    if not np.isfinite(opening[i]) or opening[i] <= 0:
                        position = 0
                        continue
                    entry_price = opening[i]
                held += 1
                gain = position * (closing[i] / entry_price - 1.0)
                if (np.isfinite(gain) and gain >= params.profit_target) or held >= params.holding_bars:
                    position = 0
                    entry_price = np.nan
                    held = 0
            elif eligible[i]:
                position = int(direction[i])
                entry_price = np.nan
                held = 0
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.bdate_range("2020-01-01", periods=13)
    data = pd.DataFrame(
        {"open": 100.0, "high": 106.0, "low": 94.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    data.loc[index[2], "close"] = 105.0
    data.loc[index[6], "close"] = 95.0
    ind = pd.DataFrame(
        {"month_edge": True, "relative_volume": 2.0, "volume_threshold": 1.0,
         "return": 0.0, "intraday_return": 0.0}, index=index,
    )
    ind.loc[index[[0, 4, 8]], ["return", "intraday_return"]] = np.array(
        [[0.01, 0.01], [-0.01, -0.01], [0.01, 0.01]]
    )
    strategy = GeneratedStrategy()
    params = MonthEdgeParams(holding_bars=3)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    actual = strategy.indicators(data, params)
    assert actual.index.equals(data.index)
    assert not strategy.generate_signals(data, actual, None, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
