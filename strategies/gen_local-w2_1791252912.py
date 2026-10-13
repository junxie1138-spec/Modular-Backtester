from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    ma_window: int = 20
    recovery_z: float = 1.5
    profit_target: float = 0.035
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791252912"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(205, params.ma_window + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if params.ma_window < 2 or params.recovery_z <= 0:
            raise ValueError("ma_window must be >= 2 and recovery_z must be positive")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["dispersion"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["distance"] = close - out["ma"]
        out["z"] = out["distance"].div(out["dispersion"].where(out["dispersion"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["previous_z"] = out["z"].shift(1)
        out["ma_step"] = out["ma"].diff()
        out["price_step"] = close.diff()
        out["distance_step"] = out["distance"].diff()
        out["regime_ma"] = close.rolling(200, min_periods=200).mean()
        out["regime_step"] = out["regime_ma"].diff(5)
        ready = out.notna().all(axis=1) & close.gt(0)
        out["long_entry"] = (
            ready
            & close.gt(out["regime_ma"])
            & out["regime_step"].gt(0)
            & out["previous_z"].le(-params.recovery_z)
            & out["z"].gt(-params.recovery_z)
            & out["z"].lt(0)
            & out["ma_step"].gt(0)
            & out["price_step"].gt(0)
            & out["distance_step"].gt(0)
        ).fillna(False)
        out["short_entry"] = (
            ready
            & close.lt(out["regime_ma"])
            & out["regime_step"].lt(0)
            & out["previous_z"].ge(params.recovery_z)
            & out["z"].lt(params.recovery_z)
            & out["z"].gt(0)
            & out["ma_step"].lt(0)
            & out["price_step"].lt(0)
            & out["distance_step"].lt(0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        if params.hold_bars < 1 or not 0 < params.profit_target < 1:
            raise ValueError("hold_bars must be positive and profit_target must be between 0 and 1")
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position:
                # The previous close's instruction fills at this bar's open.
                if entry_bar == -1:
                    entry_price = opens[i]
                    entry_bar = i
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                        continue
                gain = position * (closes[i] / entry_price - 1.0)
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                time_hit = i - entry_bar + 1 >= params.hold_bars
                if target_hit or time_hit:
                    position = 0
                    entry_price = np.nan
                    entry_bar = -1
            elif np.isfinite(closes[i]) and closes[i] > 0:
                if longs[i]:
                    position = 1
                elif shorts[i]:
                    position = -1
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=10, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 105.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[2], "close"] = 104.0
    entries = pd.DataFrame({"long_entry": False, "short_entry": False}, index=index)
    entries.loc[index[0], "long_entry"] = True
    entries.loc[index[5], "short_entry"] = True
    params = RecoveryParams(profit_target=0.03, hold_bars=3)
    result = GeneratedStrategy.generate_signals(None, data, entries, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0, 0, -1, -1, -1, 0]
    assert result.data["size"].gt(0).all()
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=250, freq="B"))
    derived = GeneratedStrategy.indicators(None, flat, RecoveryParams())
    assert not derived[["long_entry", "short_entry"]].any().any()
    assert GeneratedStrategy.warmup_bars(None, RecoveryParams()) == 205


if __name__ == "__main__":
    _self_check()
