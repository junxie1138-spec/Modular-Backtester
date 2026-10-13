from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    season_window: int = 504
    tail_percentile: float = 0.20
    profit_target: float = 0.035
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791333380"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.season_window + 5)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.season_window < 12 or not 0.0 < params.tail_percentile < 0.5:
            raise ValueError("season_window >= 12 and 0 < tail_percentile < 0.5 required")
        if params.profit_target <= 0.0 or params.max_hold < 1:
            raise ValueError("profit_target > 0 and max_hold >= 1 required")
        out = pd.DataFrame(index=data.index)
        months = data.index.year * 12 + data.index.month
        out["month_session"] = data.groupby(months, sort=False).cumcount() + 1
        # A truncated first month cannot supply a genuine fifth-session sample.
        complete_start = pd.Series(months != months[0], index=data.index) if len(data) else pd.Series(False, index=data.index)
        out["five_return"] = data["close"].pct_change(5, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["season_sample"] = out["five_return"].where(out["month_session"].eq(5) & complete_start)
        # NaNs exclude other sessions: rank only first-week monthly observations.
        # pandas Rolling.rank documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.typing.Rolling.rank.html
        out["season_rank"] = out["season_sample"].rolling(
            params.season_window, min_periods=12
        ).rank(pct=True)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        ranks = indicators["season_rank"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if side:
                # Record the actual entry open only when its bar is observable.
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        side = 0
                        continue
                gain = side * (closes[i] / entry_price - 1.0)
                # Close-observed target; exit fills at the following open.
                if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                    side = 0
                    continue
            elif i >= self.warmup_bars(params) - 1 and np.isfinite(ranks[i]):
                if ranks[i] <= params.tail_percentile:
                    side = 1
                elif ranks[i] >= 1.0 - params.tail_percentile:
                    side = -1
                if side:
                    entry_bar = i + 1
                    entry_price = np.nan
            raw[i] = side
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GeneratedParams(season_window=252, profit_target=0.02, max_hold=2)
    strategy = GeneratedStrategy()
    index = pd.bdate_range("2020-01-01", periods=270)
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    data.iloc[257, data.columns.get_loc("close")] = 103.0
    ind = pd.DataFrame({"season_rank": np.nan}, index=index)
    ind.iloc[256, 0] = 0.1
    ind.iloc[260, 0] = 0.9
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=257)
    result = strategy.generate_signals(data, ind, ctx, params).data
    expected = np.zeros(len(data), dtype=int)
    expected[257] = 1
    expected[261:263] = -1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert (result["size"] > 0.0).all()
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:240], params)
    pd.testing.assert_frame_equal(full.iloc[:240], prefix)
