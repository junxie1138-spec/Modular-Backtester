from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRetestParams:
    min_gap: float = 0.004
    shock_volume_ratio: float = 1.3
    retest_volume_ratio: float = 0.8
    profit_target: float = 0.04
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapRetestParams]):
    strategy_id = "gen_local-w2_1791263647"

    @classmethod
    def params_type(cls):
        return GapRetestParams

    def warmup_bars(self, params: GapRetestParams) -> int:
        return 10

    def indicators(self, data: pd.DataFrame, params: GapRetestParams) -> pd.DataFrame:
        if not (
            np.isfinite(params.min_gap) and params.min_gap > 0
            and np.isfinite(params.shock_volume_ratio) and params.shock_volume_ratio > 0
            and np.isfinite(params.retest_volume_ratio) and 0 < params.retest_volume_ratio < 1
            and np.isfinite(params.profit_target) and params.profit_target > 0
            and isinstance(params.max_hold_bars, int) and params.max_hold_bars > 0
        ):
            raise ValueError("Invalid gap, volume, target, or holding parameters")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["volume_baseline"] = data["volume"].rolling(8, min_periods=8).mean().shift(1)
        out["gap"] = data["open"].div(out["previous_close"].where(out["previous_close"] > 0)) - 1
        out["relative_volume"] = data["volume"].div(out["volume_baseline"].where(out["volume_baseline"] > 0))
        out["range"] = data["high"] - data["low"]
        out["close_location"] = (data["close"] - data["low"]).div(out["range"].where(out["range"] > 0))
        out["gap_midpoint"] = (data["open"] + out["previous_close"]) / 2
        out["shock"] = (
            (out["gap"] >= params.min_gap)
            & (out["relative_volume"] >= params.shock_volume_ratio)
            & (data["close"] > data["open"])
            & (out["close_location"] >= 0.7)
        ).fillna(False)
        out["prior_shock"] = out["shock"].shift(1, fill_value=False)
        out["prior_midpoint"] = out["gap_midpoint"].shift(1)
        out["prior_open"] = data["open"].shift(1)
        out["prior_close"] = data["close"].shift(1)
        out["prior_volume"] = data["volume"].shift(1)
        out["entry"] = (
            out["prior_shock"]
            & (data["low"] >= out["prior_midpoint"])
            & (data["low"] < out["prior_close"])
            & (data["close"] >= out["prior_open"])
            & (data["close"] < out["prior_close"])
            & (data["volume"] > 0)
            & (data["volume"] <= params.retest_volume_ratio * out["prior_volume"])
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRetestParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_price = 0.0
        entry_bar = -1
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_price = opens[i]
                    entry_bar = i
            if active:
                # Targets are evaluated at the close against the entry-session open.
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                time_hit = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or time_hit:
                    active = False
            elif i >= self.warmup_bars(params) - 1 and entries[i]:
                pending = True
            signals[i] = int(active or pending)
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapRetestParams()
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=40),
    )
    data.iloc[10] = [101.0, 102.2, 100.8, 102.0, 2000.0]
    data.iloc[11] = [101.8, 102.0, 101.1, 101.6, 800.0]
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=10)
    indicators = strategy.indicators(data, params)
    assert np.flatnonzero(indicators["entry"]).tolist() == [11]
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[:12].eq(0).all()
    assert result["signal"].iloc[12:30].eq(1).all()
    assert result["signal"].iloc[30:].eq(0).all()
    data.loc[data.index[14], "close"] = 105.0
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[14] == 1 and result["signal"].iloc[15] == 0
    prefix = data.iloc[:14]
    prefix_result = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
    assert result.iloc[:14].equals(prefix_result)
    assert set(result["signal"].unique()) <= {0, 1}
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
