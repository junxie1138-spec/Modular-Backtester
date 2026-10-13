from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    peak_bars: int = 20
    rate_bars: int = 3
    vol_bars: int = 20
    min_depth: float = 0.025
    rate_ratio: float = 0.6
    target_vol: float = 0.12
    profit_target: float = 0.01
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local-w2_1791246028"

    @classmethod
    def params_type(cls):
        return DrawdownParams

    def warmup_bars(self, params: DrawdownParams) -> int:
        if min(params.peak_bars, params.rate_bars, params.vol_bars) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("Holding horizon must be one or two bars.")
        if not (params.min_depth > 0 and 0 < params.rate_ratio < 1
                and params.target_vol > 0 and params.profit_target > 0):
            raise ValueError("Invalid depth, rate, volatility or target.")
        return int(max(params.peak_bars + params.rate_bars + 1,
                       params.vol_bars + 1))

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        peak = close.rolling(params.peak_bars, min_periods=params.peak_bars).max()
        depth = (1.0 - close / peak).clip(0.0, 1.0)
        susceptible = (1.0 - depth.shift(1)).clip(lower=1e-8)
        hazard = depth.diff().clip(lower=0.0) / susceptible
        baseline = hazard.shift(1).rolling(
            params.rate_bars, min_periods=params.rate_bars
        ).mean()
        ratio = hazard / baseline.where(baseline > 0)
        annual_vol = close.pct_change(fill_method=None).rolling(
            params.vol_bars, min_periods=params.vol_bars
        ).std(ddof=0) * np.sqrt(252.0)
        exhaustion = (1.0 - ratio).clip(0.0, 1.0)
        conviction = exhaustion * depth / (depth + params.min_depth)
        allocation = (params.target_vol / annual_vol.clip(lower=0.04)
                      * conviction).clip(0.05, 0.95)
        ready = (ratio.notna() & annual_vol.notna() & depth.notna()
                 & peak.eq(peak.shift(1)) & close.notna())
        entry = (ready & depth.ge(params.min_depth) & hazard.gt(0.0)
                 & ratio.le(params.rate_ratio))
        out = pd.DataFrame({
            "peak": peak,
            "depth": depth,
            "susceptible": susceptible,
            "hazard": hazard,
            "baseline": baseline,
            "rate_ratio": ratio,
            "annual_vol": annual_vol,
            "exhaustion": exhaustion,
            "conviction": conviction,
            "allocation": allocation,
            "entry": entry.astype(int),
        }, index=data.index)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["allocation"] = out["allocation"].clip(0.05, 0.95)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: DrawdownParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.05, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=bool)
        allocations = indicators["allocation"].to_numpy(dtype=float)
        active = False
        pending = False
        entry_price = 0.0
        held = 0
        position_size = 0.05
        for i in range(len(data)):
            if pending:
                # Yesterday's close decision fills at today's open.
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    active = False
                    pending = False
                    continue
                entry_price = opens[i]
                held = 0
                pending = False
            if active:
                held += 1
                target_hit = (np.isfinite(closes[i])
                              and closes[i] >= entry_price * (1.0 + params.profit_target))
                if target_hit or held >= params.max_hold_bars:
                    # Exit decisions execute at the following open after shifting.
                    active = False
                    continue
                signals[i] = 1
                sizes[i] = position_size
            elif entries[i]:
                active = True
                pending = True
                position_size = float(np.clip(allocations[i], 0.05, 0.95))
                signals[i] = 1
                sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = DrawdownParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100.0] * 5, "high": [102.0] * 5,
        "low": [99.0] * 5, "close": [100.0] * 5,
        "volume": [1000.0] * 5,
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [1, 0, 0, 0, 0], "allocation": [0.3] * 5,
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    assert result["size"].iloc[1] == 0.3
    data.loc[index[1], "close"] = 102.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0]
    derived = GeneratedStrategy.indicators(None, data, params)
    assert derived["entry"].eq(0).all()
    assert np.isfinite(derived.to_numpy()).all()


if __name__ == "__main__":
    _self_check()
