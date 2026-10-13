from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalRecoveryParams:
    complexity_jump: float = 0.08
    stretch: float = 0.012
    profit_target: float = 0.04
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[OrdinalRecoveryParams]):
    strategy_id = "gen_local-w2_1791287984"

    @classmethod
    def params_type(cls):
        return OrdinalRecoveryParams

    def warmup_bars(self, params: OrdinalRecoveryParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: OrdinalRecoveryParams) -> pd.DataFrame:
        if not (0.0 < params.complexity_jump <= 1.0):
            raise ValueError("complexity_jump must be in (0, 1]")
        if not (0.0 < params.stretch < 1.0):
            raise ValueError("stretch must be in (0, 1)")
        if not (0.0 < params.profit_target < 1.0):
            raise ValueError("profit_target must be in (0, 1)")
        if not isinstance(params.holding_bars, int) or params.holding_bars < 1:
            raise ValueError("holding_bars must be a positive integer")
        close = data["close"].astype(float).where(lambda s: np.isfinite(s) & s.gt(0))
        out = pd.DataFrame(index=data.index)
        out["close_clean"] = close
        out["close_lag1"] = close.shift(1)
        out["close_lag2"] = close.shift(2)
        a, b = out["close_lag2"], out["close_lag1"]
        valid = a.notna() & b.notna() & close.notna()
        # Equal prices retain chronological order, giving deterministic ordinal patterns.
        out["pattern"] = ((a > b).astype(int) * 4 + (a > close).astype(int) * 2 + (b > close).astype(int)).where(valid)
        frequency_columns = []
        for code in (0, 1, 3, 4, 6, 7):
            name = f"frequency_{code}"
            out[name] = out["pattern"].eq(code).where(valid).rolling(6, min_periods=6).mean()
            frequency_columns.append(name)
        frequencies = out[frequency_columns]
        out["complexity"] = -(frequencies * np.log(frequencies.where(frequencies > 0, 1.0))).sum(axis=1, min_count=6) / np.log(6.0)
        out["complexity_previous"] = out["complexity"].shift(1)
        out["complexity_change"] = out["complexity"] - out["complexity_previous"]
        out["peak"] = close.rolling(8, min_periods=8).max()
        out["trough"] = close.rolling(8, min_periods=8).min()
        out["drawdown"] = close / out["peak"] - 1.0
        out["runup"] = close / out["trough"] - 1.0
        out["fragmenting"] = out["complexity"].ge(0.5) & out["complexity_change"].ge(params.complexity_jump)
        out["long_entry"] = (out["fragmenting"] & out["drawdown"].le(-params.stretch) & close.gt(b)).fillna(False)
        out["short_entry"] = (out["fragmenting"] & out["runup"].ge(params.stretch) & close.lt(b)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalRecoveryParams) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # The pending entry becomes observable only on its fill bar.
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                if position:
                    gain = position * (closes[i] / entry_price - 1.0)
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.holding_bars:
                        position = 0
            elif i >= self.warmup_bars(params) - 1 and (longs[i] or shorts[i]):
                position = 1 if longs[i] else -1
                entry_bar = i + 1
                entry_price = np.nan
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = OrdinalRecoveryParams()
    index = pd.date_range("2020-01-01", periods=30, freq="B")
    for side in (1, -1):
        for hit_target in (False, True):
            close = np.full(30, 100.0)
            close[9] = 90.0 if side == 1 else 110.0
            if hit_target:
                close[12] = 105.0 if side == 1 else 95.0
            data = pd.DataFrame({"open": 100.0, "high": np.maximum(close, 100.0), "low": np.minimum(close, 100.0), "close": close, "volume": 1000.0}, index=index)
            derived = strategy.indicators(data, params)
            assert derived["complexity"].first_valid_index() == index[7]
            assert derived["complexity_previous"].first_valid_index() == index[8]
            derived["long_entry"] = False
            derived["short_entry"] = False
            derived.loc[index[9], "long_entry" if side == 1 else "short_entry"] = True
            result = strategy.generate_signals(data, derived, None, params).data
            end = 13 if hit_target else 28
            assert result["signal"].iloc[:10].eq(0).all()
            assert result["signal"].iloc[10:end].eq(side).all()
            assert result["signal"].iloc[end:].eq(0).all()
            assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
