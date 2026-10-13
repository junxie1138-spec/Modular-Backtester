from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TransmissionParams:
    rank_window: int = 4
    entry_percentile: float = 0.8
    min_autocorr: float = 0.25


class GeneratedStrategy(BaseStrategy[TransmissionParams]):
    strategy_id = "gen_local_1791290437"

    @classmethod
    def params_type(cls) -> type[TransmissionParams]:
        return TransmissionParams

    def warmup_bars(self, params: TransmissionParams) -> int:
        if not 3 <= params.rank_window <= 5:
            raise ValueError("rank_window must be between 3 and 5")
        if not 0.5 < params.entry_percentile <= 1.0:
            raise ValueError("entry_percentile must be in (0.5, 1]")
        if not 0.0 <= params.min_autocorr < 1.0:
            raise ValueError("min_autocorr must be in [0, 1)")
        # Return, lag, correlation, rank, and confirmation require n + 5 bars.
        return int(params.rank_window + 5)

    def indicators(self, data: pd.DataFrame, params: TransmissionParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lagged_return"] = out["return"].shift(1)
        out["autocorr"] = out["return"].rolling(3).corr(out["lagged_return"]).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        out["net_return"] = out["return"].rolling(3).sum()
        out["absolute_return"] = out["return"].abs().rolling(3).sum()
        out["efficiency"] = out["net_return"].div(out["absolute_return"].replace(0.0, np.nan))
        out["transmission"] = out["autocorr"].clip(lower=0.0) * out["efficiency"]
        # Rolling.rank returns the latest observation's trailing percentile.
        # Documentation: [pandas Rolling.rank](https://pandas.pydata.org/docs/reference/api/pandas.api.typing.Rolling.rank.html)
        out["raw_percentile"] = out["transmission"].rolling(params.rank_window).rank(pct=True)
        floor = 1.0 / params.rank_window
        out["percentile"] = ((out["raw_percentile"] - floor) / (1.0 - floor)).clip(0.0, 1.0)
        eligible = out["autocorr"].ge(params.min_autocorr)
        out["long_candidate"] = eligible & out["transmission"].gt(0.0) & out["percentile"].ge(params.entry_percentile)
        out["short_candidate"] = eligible & out["transmission"].lt(0.0) & out["percentile"].le(1.0 - params.entry_percentile)
        out["long_confirmed"] = out["long_candidate"].rolling(2).sum().eq(2.0)
        out["short_confirmed"] = out["short_candidate"].rolling(2).sum().eq(2.0)
        out["entry"] = np.select([out["long_confirmed"], out["short_confirmed"]], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: TransmissionParams) -> SignalFrame:
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = TransmissionParams()
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    entries = pd.DataFrame({"entry": [0, 1, 0, 0, -1, -1, 0, 1, 1]}, index=index)
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    flat = strategy.indicators(data, params)
    assert flat["entry"].eq(0).all()
    for window in (3, 4, 5):
        assert strategy.warmup_bars(TransmissionParams(rank_window=window)) <= 10
