from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    target_vol: float = 0.12
    profit_target: float = 0.01


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791255346"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 34

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        weights = np.ones(32)
        for k in range(1, 32):
            weights[k] = -weights[k - 1] * (0.4 - k + 1) / k
        fractional = sum(weights[k] * close.shift(k) for k in range(32))
        returns = close.pct_change(fill_method=None)
        daily_vol = returns.rolling(20, min_periods=20).std()
        innovation = fractional.diff().div(close.shift(1)).div(
            daily_vol.replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        backlog = np.full(len(data), np.nan)
        overflow = np.full(len(data), np.nan)
        queue = 0.0
        for i, value in enumerate(innovation.to_numpy()):
            if not np.isfinite(value):
                queue = 0.0
                continue
            incoming = 0.8 * queue + value
            queue = float(np.clip(incoming, -2.0, 2.0))
            backlog[i] = queue
            overflow[i] = incoming - queue
        out = pd.DataFrame({
            "fractional_close": fractional,
            "daily_vol": daily_vol,
            "annual_vol": daily_vol * np.sqrt(252.0),
            "innovation": innovation,
            "backlog": backlog,
            "overflow": overflow,
        }, index=data.index)
        fading = innovation.abs().lt(innovation.shift(1).abs())
        newly_full = out["overflow"].shift(1).abs().le(0.15)
        trigger = out["overflow"].abs().gt(0.15) & fading & newly_full
        out["entry"] = np.where(
            trigger, -np.sign(out["overflow"].fillna(0.0)), 0
        ).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        if not (np.isfinite(params.target_vol) and params.target_vol > 0.0):
            raise ValueError("target_vol must be positive and finite")
        if not (np.isfinite(params.profit_target) and params.profit_target > 0.0):
            raise ValueError("profit_target must be positive and finite")
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.95, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        vols = indicators["annual_vol"].to_numpy(dtype=float)
        side, entry_bar, entry_price, held_size = 0, -1, np.nan, 0.95
        for i in range(n):
            if side:
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        side = 0
                gain = side * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or i - entry_bar + 1 >= 4:
                    side = 0
            elif i >= self.warmup_bars(params) and entries[i]:
                if np.isfinite(vols[i]) and vols[i] > 0.0:
                    side = int(entries[i])
                    entry_bar = i + 1
                    entry_price = np.nan
                    held_size = float(np.clip(params.target_vol / vols[i], 0.05, 0.95))
            signals[i] = side
            sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy, params = GeneratedStrategy(), Params()
    index = pd.date_range("2020-01-01", periods=48, freq="B")
    data = pd.DataFrame({
        "open": 100.0, "high": 103.0, "low": 97.0,
        "close": 100.0, "volume": 1000000.0,
    }, index=index)
    indicators = strategy.indicators(data, params)
    assert not strategy.generate_signals(data, indicators, None, params).data["signal"].any()
    indicators["entry"] = 0
    indicators["annual_vol"] = 0.2
    indicators.iloc[34, indicators.columns.get_loc("entry")] = 1
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].iloc[34] == 0
    assert result["signal"].iloc[35:39].eq(1).all()
    assert result["signal"].iloc[39] == 0
    assert np.isclose(result["size"].iloc[35], 0.6)
    for direction in (1, -1):
        indicators.iloc[34, indicators.columns.get_loc("entry")] = direction
        data.loc[index[36], "close"] = 100.0 * (1.0 + direction * 0.02)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].iloc[35:37].eq(direction).all()
        assert result["signal"].iloc[37] == 0
        assert result["size"].gt(0.0).all()
