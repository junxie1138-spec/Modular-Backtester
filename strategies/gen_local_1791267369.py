from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    fast_bars: int = 3
    slow_bars: int = 12
    regime_bars: int = 20
    percentile_bars: int = 126
    shock_quantile: float = 0.85
    regime_quantile: float = 0.60
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local_1791267369"

    @classmethod
    def params_type(cls):
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        return int(max(
            2 * params.fast_bars + params.percentile_bars,
            params.slow_bars + params.fast_bars,
            params.regime_bars + params.percentile_bars,
        ) + 1)

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["price_change"] = close.diff()
        out["fast_roc"] = close.pct_change(params.fast_bars, fill_method=None)
        out["slow_roc"] = close.pct_change(params.slow_bars, fill_method=None)
        out["fast_acceleration"] = out["fast_roc"].diff(params.fast_bars) / params.fast_bars
        out["slow_acceleration"] = out["slow_roc"].diff(params.fast_bars) / params.slow_bars
        out["shock_threshold"] = out["fast_acceleration"].abs().rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.shock_quantile).shift(1)
        out["path_length"] = out["price_change"].abs().rolling(
            params.regime_bars, min_periods=params.regime_bars
        ).sum()
        out["displacement"] = close.diff(params.regime_bars).abs()
        out["efficiency"] = out["displacement"] / out["path_length"].replace(0.0, np.nan)
        out["regime_threshold"] = out["efficiency"].rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.regime_quantile).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveParams,
    ) -> SignalFrame:
        needed = indicators[[
            "fast_acceleration", "slow_acceleration", "shock_threshold",
            "efficiency", "regime_threshold",
        ]]
        ready = np.isfinite(needed.to_numpy()).all(axis=1)
        fast = indicators["fast_acceleration"]
        slow = indicators["slow_acceleration"]
        shock = fast.abs() > indicators["shock_threshold"]
        flowing = indicators["efficiency"] > indicators["regime_threshold"]
        transmitted = ready & shock & flowing & (fast * slow > 0)
        reflected = ready & shock & ~flowing & (fast * slow < 0)
        entries = np.select(
            [transmitted, reflected], [np.sign(fast), -np.sign(fast)], default=0
        ).astype(int)
        target = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for bar, entry in enumerate(entries):
            if position != 0 and bar == exit_bar:
                position = 0
            elif position == 0 and entry != 0:
                position = int(entry)
                exit_bar = bar + params.hold_bars
            target[bar] = position
        df = pd.DataFrame({"signal": target, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = ShockwaveParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=params.hold_bars + 3, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    ind = pd.DataFrame({
        "fast_acceleration": 0.02, "slow_acceleration": 0.01,
        "shock_threshold": 0.005, "efficiency": 0.8, "regime_threshold": 0.5,
    }, index=index)
    for fast, slow, efficiency, expected in [
        (0.02, 0.01, 0.8, 1), (-0.02, -0.01, 0.8, -1),
        (0.02, -0.01, 0.2, -1), (-0.02, 0.01, 0.2, 1),
    ]:
        ind["fast_acceleration"] = fast
        ind["slow_acceleration"] = slow
        ind["efficiency"] = efficiency
        signal = strategy.generate_signals(data, ind, ctx, params).data["signal"]
        assert signal.iloc[0] == 0
        assert (signal.iloc[1:params.hold_bars + 1] == expected).all()
        assert signal.iloc[params.hold_bars + 1] == 0
        assert signal.iloc[params.hold_bars + 2] == expected
    ind.loc[:, :] = np.nan
    assert (strategy.generate_signals(data, ind, ctx, params).data["signal"] == 0).all()
