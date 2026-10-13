from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    ma_bars: int = 20
    percentile_bars: int = 126
    entry_percentile: float = 0.80
    profit_target: float = 0.015
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local_1790859785"

    @classmethod
    def params_type(cls):
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        return int(params.ma_bars + params.percentile_bars + 3)

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        if params.ma_bars < 2 or params.percentile_bars < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must lie between zero and one.")
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        out = pd.DataFrame(index=data.index)
        window = close.rolling(params.ma_bars, min_periods=params.ma_bars)
        out["ma"] = window.mean()
        out["price_std"] = window.std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["price_std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["threshold"] = out["z"].shift(1).rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.entry_percentile)
        out["ma_rise"] = out["ma"].diff(3)
        z = out["z"]
        out["contraction"] = (
            (z.shift(3) > z.shift(2))
            & (z.shift(2) > z.shift(1))
            & (z.shift(1) > 0)
        )
        out["entry"] = (
            out["contraction"]
            & (out["ma_rise"] > 0)
            & (z > 0)
            & (z > out["threshold"])
            & (z.shift(1) <= out["threshold"].shift(1))
            & (z > z.shift(1))
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveParams,
    ) -> SignalFrame:
        if not 3 <= params.hold_bars <= 5 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Use a positive profit target and a three-to-five-bar time stop.")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if active:
                # The preceding close's entry decision fills at this bar's open.
                if entry_bar == i:
                    entry_price = opens[i]
                target_hit = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if target_hit or i - entry_bar + 1 >= params.hold_bars:
                    active = False
                else:
                    signal[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = i + 1
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = WaveParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame({"entry": [True] + [False] * 7}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    # A gap changes the entry basis; the target is measured from the fill open.
    data.loc[index[1], ["open", "close"]] = 102.0
    data.loc[index[2], "close"] = 104.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert (result["size"] > 0).all()
    assert not strategy.indicators(data, params)["entry"].any()


if __name__ == "__main__":
    _self_check()
