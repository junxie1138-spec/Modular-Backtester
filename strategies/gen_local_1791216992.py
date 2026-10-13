from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    pattern_window: int = 12
    baseline_window: int = 24
    entropy_drop: float = 0.10
    up_share: float = 0.35
    atr_period: int = 14
    stop_atr: float = 1.8
    hold_bars: int = 4

    def __post_init__(self):
        if min(self.pattern_window, self.baseline_window, self.atr_period) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (0.0 <= self.entropy_drop <= 1.0 and 0.0 < self.up_share <= 1.0):
            raise ValueError("Invalid entropy or frequency threshold.")
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0 or not 3 <= self.hold_bars <= 5:
            raise ValueError("Require positive stop_atr and a three-to-five-bar hold.")


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791216992"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.pattern_window + params.baseline_window + 4, params.atr_period + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        c = data["close"]
        a, b = c.shift(2), c.shift(1)
        # Six strict chronological rank permutations; ties are unclassified.
        out["pattern"] = np.select(
            [
                (a < b) & (b < c),
                (a < c) & (c < b),
                (b < a) & (a < c),
                (b < c) & (c < a),
                (c < a) & (a < b),
                (c < b) & (b < a),
            ],
            [0, 1, 2, 3, 4, 5],
            default=np.nan,
        )
        probability_columns = []
        for code in range(6):
            name = f"p{code}"
            probability_columns.append(name)
            out[name] = (
                out["pattern"].eq(code).astype(float)
                .where(out["pattern"].notna())
                .rolling(params.pattern_window, min_periods=params.pattern_window).mean()
            )
        probabilities = out[probability_columns]
        terms = probabilities * np.log(probabilities.where(probabilities > 0, 1.0))
        out["entropy"] = -terms.sum(axis=1, min_count=6) / np.log(6.0)
        out["baseline_entropy"] = out["entropy"].shift(1).rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean()
        out["entropy_contraction"] = out["baseline_entropy"] - out["entropy"]
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - b).abs(), (data["low"] - b).abs()],
            axis=1,
        ).max(axis=1).where(b.notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        out["eligible"] = (
            (out["entropy_contraction"] >= params.entropy_drop)
            & (out["p0"] >= params.up_share)
            & (out["p0"] > out["p5"])
            & (c > b)
            & out["atr"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        eligible = indicators["eligible"].fillna(False).astype(bool)
        confirmed = (eligible & eligible.shift(1, fill_value=False)).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # Entry fills at this open; freeze the preceding decision bar's ATR.
                    if not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                        entry_bar = -1
                        continue
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                if not np.isfinite(closes[i]) or closes[i] < stop or i - entry_bar + 1 >= params.hold_bars:
                    entry_bar = -1
                else:
                    raw[i] = 1
            elif confirmed[i]:
                raw[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 112.0, "low": 95.0,
        "close": [100.0, 100.0, 100.0, 110.0, 99.0, 97.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "eligible": [True, True, False, False, False, False, False, False],
        "atr": 1.0,
    }, index=index)
    params = GeneratedParams(stop_atr=2.0, hold_bars=5)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    params = GeneratedParams(stop_atr=2.0, hold_bars=3)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    indicators["eligible"] = [True, False, False, False, False, False, False, False]
    assert not strategy.generate_signals(data, indicators, ctx, params).data["signal"].any()
    assert strategy.indicators(data.iloc[:0], params).empty


if __name__ == "__main__":
    _self_check()
