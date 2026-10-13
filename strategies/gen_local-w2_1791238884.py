from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalBreakoutParams:
    entropy_window: int = 24
    breakout_window: int = 20
    complexity_lag: int = 5
    entropy_ceiling: float = 0.80
    entropy_drop: float = 0.05
    profit_target: float = 0.03
    holding_bars: int = 8

    def __post_init__(self):
        if min(self.entropy_window, self.breakout_window, self.complexity_lag, self.holding_bars) < 1:
            raise ValueError("Lookbacks and holding_bars must be positive.")
        if not 0.0 < self.entropy_ceiling <= 1.0:
            raise ValueError("entropy_ceiling must be in (0, 1].")
        if not 0.0 <= self.entropy_drop <= 1.0:
            raise ValueError("entropy_drop must be in [0, 1].")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0.0:
            raise ValueError("profit_target must be finite and positive.")


class GeneratedStrategy(BaseStrategy[OrdinalBreakoutParams]):
    strategy_id = "gen_local-w2_1791238884"

    @classmethod
    def params_type(cls):
        return OrdinalBreakoutParams

    def warmup_bars(self, params):
        return int(max(params.entropy_window + 2 + params.complexity_lag,
                       params.breakout_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        triples = np.column_stack((data["close"].shift(2),
                                   data["close"].shift(1), data["close"]))
        valid = np.isfinite(triples).all(axis=1)
        # Equal prices are ranked in chronological order.
        order = np.argsort(triples, axis=1, kind="stable")
        codes = 9 * order[:, 0] + 3 * order[:, 1] + order[:, 2]
        out["pattern"] = pd.Series(codes, index=data.index, dtype=float).where(valid)
        frequencies = []
        for code in (5, 7, 11, 15, 19, 21):
            name = f"frequency_{code}"
            occurrence = out["pattern"].eq(code).astype(float).where(valid)
            out[name] = occurrence.rolling(params.entropy_window,
                                           min_periods=params.entropy_window).mean()
            frequencies.append(name)
        probabilities = out[frequencies]
        out["entropy"] = -(probabilities * np.log(probabilities.clip(lower=1e-12))).sum(
            axis=1, min_count=6) / np.log(6.0)
        out["lagged_entropy"] = out["entropy"].shift(params.complexity_lag)
        out["prior_high"] = data["high"].rolling(
            params.breakout_window, min_periods=params.breakout_window).max().shift(1)
        out["ordinal_agreement"] = (
            out["entropy"].le(params.entropy_ceiling)
            & (out["lagged_entropy"] - out["entropy"]).ge(params.entropy_drop)
        ).fillna(False)
        out["price_agreement"] = data["close"].gt(out["prior_high"]).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        entries = (indicators["ordinal_agreement"].fillna(False)
                   & indicators["price_agreement"].fillna(False)).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # Observe the entry open only when that bar has arrived.
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0.0:
                    entry_bar = -1
                    continue
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (
                    1.0 + params.profit_target)
                time_expired = i - entry_bar + 1 >= params.holding_bars
                if target_hit or time_expired:
                    entry_bar = -1
                    entry_price = np.nan
                    continue
                decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                entry_bar = i + 1
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = OrdinalBreakoutParams(entropy_window=3, breakout_window=3,
                                  complexity_lag=1, holding_bars=3)
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({
        "open": [100, 102, 100, 100, 100, 100, 100, 100, 100],
        "high": [101, 105, 107, 101, 101, 101, 101, 101, 101],
        "low": [99] * 9,
        "close": [100, 104, 106, 100, 100, 100, 100, 100, 100],
        "volume": [1000] * 9,
    }, index=index)
    gates = pd.DataFrame({
        "ordinal_agreement": [True, False, True, False, True, False, False, False, False],
        "price_agreement": [True, False, False, True, True, False, False, False, False],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d",
                          warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, gates, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 1, 1, 1, 0]
    assert result["size"].gt(0.0).all()
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:7], params)
    pd.testing.assert_frame_equal(full.iloc[:7], prefix)
    assert full["entropy"].dropna().between(-1e-12, 1.0 + 1e-12).all()
    assert full["entropy"].iloc[:4].isna().all()


if __name__ == "__main__":
    _self_check()
