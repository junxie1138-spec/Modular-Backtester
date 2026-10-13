from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDrawdownParams:
    peak_window: int = 20
    min_drawdown: float = 0.025
    gap_share: float = 0.20
    profit_target: float = 0.015
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[GapDrawdownParams]):
    strategy_id = "gen_local_1791231639"

    @classmethod
    def params_type(cls):
        return GapDrawdownParams

    def warmup_bars(self, params: GapDrawdownParams) -> int:
        if params.peak_window < 2 or params.max_hold < 1:
            raise ValueError("Lookback must be >= 2 and holding period >= 1.")
        if not (0 < params.min_drawdown < 1 and 0 < params.gap_share < 1):
            raise ValueError("Drawdown and gap share must lie between zero and one.")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Profit target must be finite and positive.")
        return int(max(200, params.peak_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapDrawdownParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["prior_peak"] = data["high"].rolling(params.peak_window, min_periods=params.peak_window).max().shift(1)
        out["prior_close"] = data["close"].shift(1)
        out["gap_down"] = out["prior_close"] - data["open"]
        out["open_depth"] = out["prior_peak"] - data["open"]
        out["open_drawdown"] = out["open_depth"] / out["prior_peak"].where(out["prior_peak"] > 0)
        out["gap_share"] = out["gap_down"] / out["open_depth"].where(out["open_depth"] > 0)
        out["gap_repair"] = (data["close"] - data["open"]) / out["gap_down"].where(out["gap_down"] > 0)
        out["entry"] = (
            (data["close"] > out["ma200"])
            & (out["open_drawdown"] >= params.min_drawdown)
            & (out["gap_share"] >= params.gap_share)
            & (out["gap_down"] > 0)
            & out["gap_repair"].between(0.25, 0.75)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapDrawdownParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signal = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                # The previous close's entry decision fills at this open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i])
                # Close-based exits also execute at the following open.
                if not valid or closes[i] >= entry_price * (1 + params.profit_target) or i - entry_bar + 1 >= params.max_hold:
                    active = False
            elif entries[i]:
                active = True
                entry_bar = i + 1
            signal[i] = int(active)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapDrawdownParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({name: 100.0 for name in ("open", "high", "low", "close", "volume")}, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, False, False, False]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    data.loc[index[2], "close"] = 102.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) == 200
    assert not strategy.indicators(data, params)["entry"].any()


if __name__ == "__main__":
    _self_check()
