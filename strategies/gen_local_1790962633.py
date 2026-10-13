from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    window: int = 6
    entropy_ceiling: float = 0.68
    profit_target: float = 0.025
    hold_bars: int = 18

    def __post_init__(self):
        if not 3 <= self.window <= 7:
            raise ValueError("window must be between 3 and 7")
        if not 0.0 < self.entropy_ceiling <= 1.0:
            raise ValueError("entropy_ceiling must be in (0, 1]")
        if not 0.0 < self.profit_target < 1.0:
            raise ValueError("profit_target must be in (0, 1)")
        if not 15 <= self.hold_bars <= 20:
            raise ValueError("hold_bars must be between 15 and 20")


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = "gen_local_1790962633"

    @classmethod
    def params_type(cls):
        return OrdinalParams

    def warmup_bars(self, params: OrdinalParams) -> int:
        return params.window + 3

    def indicators(self, data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lag1"] = out["return"].shift(1)
        out["lag2"] = out["return"].shift(2)
        a, b, c = out["lag2"], out["lag1"], out["return"]
        # Tied or missing triples have no ordinal pattern.
        out["pattern"] = np.select(
            [(a < b) & (b < c), (a < c) & (c < b),
             (b < a) & (a < c), (b < c) & (c < a),
             (c < a) & (a < b), (c < b) & (b < a)],
            [0, 1, 2, 3, 4, 5], default=np.nan,
        )
        terms = pd.DataFrame(index=data.index)
        for k in range(6):
            probability = (out["pattern"] == k).astype(float).where(out["pattern"].notna())
            out[f"probability_{k}"] = probability.rolling(params.window, min_periods=params.window).mean()
            p = out[f"probability_{k}"]
            terms[k] = -p * np.log(p.clip(lower=1e-12))
        out["entropy"] = terms.sum(axis=1, min_count=6) / np.log(6.0)
        out["autocorrelation"] = (
            out["return"].rolling(params.window, min_periods=params.window)
            .corr(out["lag1"]).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        )
        out["forecast"] = out["return"] * out["autocorrelation"]
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: OrdinalParams,
    ) -> SignalFrame:
        ready = (
            indicators["entropy"].le(params.entropy_ceiling)
            & indicators["autocorrelation"].abs().ge(0.15)
            & indicators["forecast"].notna()
        )
        candidate = np.sign(indicators["forecast"].where(ready, 0.0)).fillna(0.0).astype(int).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        position, entry_bar, entry_price = 0, -1, np.nan
        for i in range(len(data)):
            if position:
                # The previous close's decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                else:
                    gain = position * (closes[i] / entry_price - 1.0)
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                        position = 0
            elif candidate[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(candidate[i])
                entry_bar, entry_price = i + 1, np.nan
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), OrdinalParams()
    index = pd.date_range("2020-01-01", periods=23, freq="B")
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    quiet = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert quiet["signal"].eq(0).all() and strategy.warmup_bars(params) <= 10
    forced = pd.DataFrame({"entropy": 0.1, "autocorrelation": 0.5, "forecast": 0.01}, index=index)
    timed = strategy.generate_signals(data, forced, ctx, params).data
    assert timed["signal"].iloc[0] == 0
    assert timed["signal"].iloc[1:19].eq(1).all() and timed["signal"].iloc[19] == 0
    for side in (1, -1):
        moved = data.copy()
        moved.loc[index[1], "close"] = 100.0 * (1.0 + side * 0.03)
        forced["forecast"] = side * 0.01
        result = strategy.generate_signals(moved, forced, ctx, params).data
        assert result["signal"].iloc[1] == side and result["signal"].iloc[2] == 0


if __name__ == "__main__":
    _self_check()
