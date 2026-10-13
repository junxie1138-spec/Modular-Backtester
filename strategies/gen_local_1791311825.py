from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    peak_window: int = 40
    impact_window: int = 4
    drawdown_min: float = 0.04
    impact_ratio_max: float = 0.55
    profit_target: float = 0.025
    max_hold: int = 8
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791311825"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.peak_window, 2 * params.impact_window + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"].gt(0))
        volume = data["volume"].where(data["volume"].gt(0))
        out["peak"] = close.rolling(params.peak_window).max()
        out["drawdown"] = 1.0 - close / out["peak"]
        out["return"] = close.pct_change(fill_method=None)
        out["loss"] = (-out["return"]).clip(lower=0)
        out["down_volume"] = volume.where(out["return"].lt(0), 0.0).where(
            out["return"].notna() & volume.notna()
        )
        w = params.impact_window
        out["loss_sum"] = out["loss"].rolling(w).sum()
        out["down_volume_sum"] = out["down_volume"].rolling(w).sum()
        out["impact"] = out["loss_sum"] / out["down_volume_sum"].where(
            out["down_volume_sum"].gt(0)
        )
        out["prior_impact"] = out["impact"].shift(w)
        out["impact_ratio"] = out["impact"] / out["prior_impact"].where(
            out["prior_impact"].gt(0)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        eligible = (
            indicators["drawdown"].ge(params.drawdown_min)
            & indicators["impact_ratio"].le(params.impact_ratio_max)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                # Record the fill only when its opening bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                target_hit = (
                    np.isfinite(entry_price)
                    and entry_price > 0
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                timed_out = i - entry_bar + 1 >= params.max_hold
                if target_hit or timed_out:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    raw[i] = 1
            elif eligible[i] and np.isfinite(closes[i]) and closes[i] > 0:
                entry_bar = i + 1
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams(peak_window=2, impact_window=1, max_hold=3)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=3)
    ind = pd.DataFrame({"drawdown": np.nan, "impact_ratio": np.nan}, index=index)
    ind.iloc[2] = [0.05, 0.4]
    for target, expected in (
        (False, [0, 0, 0, 1, 1, 1, 0, 0]),
        (True, [0, 0, 0, 1, 0, 0, 0, 0]),
    ):
        data.loc[index[3], "close"] = 103.0 if target else 100.0
        signals = strategy.generate_signals(data, ind, ctx, params).data
        assert signals["signal"].tolist() == expected
        assert signals["size"].gt(0).all()
    for column, value in (("drawdown", 0.0), ("impact_ratio", 2.0)):
        blocked = ind.copy()
        blocked.loc[index[2], column] = value
        assert not strategy.generate_signals(data, blocked, ctx, params).data["signal"].any()
    flat = data.copy()
    flat["close"] = 100.0
    assert not strategy.generate_signals(
        flat, strategy.indicators(flat, params), ctx, params
    ).data["signal"].any()


if __name__ == "__main__":
    _self_check()
