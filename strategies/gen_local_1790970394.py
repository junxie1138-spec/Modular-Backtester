from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    energy_window: int = 16
    correlation_window: int = 32
    percentile_window: int = 126
    energy_percentile: float = 0.75
    correlation_percentile: float = 0.25
    profit_target: float = 0.012
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local_1790970394"

    @classmethod
    def params_type(cls):
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        return int(max(8 + params.energy_window, 3 + params.correlation_window)
                   + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        if min(params.energy_window, params.correlation_window,
               params.percentile_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (0 < params.energy_percentile < 1
                and 0 < params.correlation_percentile < 1):
            raise ValueError("Percentiles must be between zero and one.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        r = out["return"]
        for scale in (2, 4, 8):
            half = scale // 2
            block = r.rolling(half, min_periods=half).sum()
            coefficient = (block - block.shift(half)) / np.sqrt(scale)
            out[f"haar_{scale}"] = coefficient
            out[f"energy_{scale}"] = coefficient.pow(2).rolling(
                params.energy_window, min_periods=params.energy_window).mean()
        total = out[["energy_2", "energy_4", "energy_8"]].sum(axis=1, min_count=3)
        out["resonance"] = out["energy_4"] / total.where(total > 0)
        out["resonance_threshold"] = out["resonance"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.energy_percentile)
        out["lag_two_correlation"] = r.rolling(
            params.correlation_window, min_periods=params.correlation_window
        ).corr(r.shift(2)).replace([np.inf, -np.inf], np.nan)
        out["correlation_threshold"] = out["lag_two_correlation"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.correlation_percentile)
        out["phase_forecast"] = out["lag_two_correlation"] * r.shift(1)
        gate = (
            (out["resonance"] > out["resonance_threshold"])
            & (out["lag_two_correlation"] < out["correlation_threshold"])
            & (out["lag_two_correlation"] < 0)
            & np.isfinite(out["phase_forecast"])
        )
        out["entry"] = np.sign(out["phase_forecast"].fillna(0)).where(gate, 0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: WaveParams,
    ) -> SignalFrame:
        if params.profit_target <= 0 or not 3 <= params.hold_bars <= 5:
            raise ValueError("Use a positive profit target and a three-to-five-bar hold.")
        desired = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(0).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        side, entry_bar, entry_price = 0, -1, np.nan
        for i in range(len(data)):
            if side:
                # Capture the next open only when that bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    side = 0
                else:
                    gain = side * (closes[i] / entry_price - 1.0)
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                        side = 0
                # An exit always leaves a flat bar before another entry.
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(entries[i])
                entry_bar, entry_price = i + 1, np.nan
            desired[i] = side
        df = pd.DataFrame({"signal": desired, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 97.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    data.loc[index[2], "close"] = 102.0
    data.loc[index[8], "close"] = 98.0
    entries = pd.DataFrame({"entry": [1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    signals = strategy.generate_signals(data, entries, ctx, WaveParams()).data
    assert signals["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, -1, -1, 0, 0, 0]
    timed = strategy.generate_signals(data, entries, ctx, WaveParams(profit_target=1.0)).data
    assert timed["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, 0]
    assert (signals["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
