from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapEnergyParams:
    energy_window: int = 16
    drift_window: int = 8
    coherence_threshold: float = 0.45
    profit_target: float = 0.06
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[GapEnergyParams]):
    strategy_id = "gen_local_1790984705"

    @classmethod
    def params_type(cls):
        return GapEnergyParams

    def warmup_bars(self, params: GapEnergyParams) -> int:
        return max(8 + params.energy_window, params.drift_window + 1) + 1

    def indicators(self, data: pd.DataFrame, params: GapEnergyParams) -> pd.DataFrame:
        if params.energy_window < 2 or params.drift_window < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 0.0 < params.coherence_threshold <= 1.0:
            raise ValueError("Coherence threshold must be in (0, 1]")
        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        out["gap"] = np.log(opening / closing.shift(1))
        out["intraday"] = np.log(closing / opening)
        out["close_return"] = np.log(closing / closing.shift(1))
        gap_energy = pd.Series(0.0, index=data.index)
        day_energy = pd.Series(0.0, index=data.index)
        cross_energy = pd.Series(0.0, index=data.index)
        for half in (1, 2, 4):
            for column in ("gap", "intraday"):
                out[f"{column}_mean_{half}"] = out[column].rolling(half, min_periods=half).mean()
                block = out[f"{column}_mean_{half}"]
                out[f"{column}_haar_{half}"] = (
                    (block - block.shift(half)) * np.sqrt(half / 2.0)
                )
            g = out[f"gap_haar_{half}"]
            d = out[f"intraday_haar_{half}"]
            out[f"gap_energy_{half}"] = g.pow(2).rolling(params.energy_window).mean()
            out[f"day_energy_{half}"] = d.pow(2).rolling(params.energy_window).mean()
            out[f"cross_energy_{half}"] = (g * d).rolling(params.energy_window).mean()
            gap_energy = gap_energy + out[f"gap_energy_{half}"]
            day_energy = day_energy + out[f"day_energy_{half}"]
            cross_energy = cross_energy + out[f"cross_energy_{half}"]
        out["gap_energy"] = gap_energy
        out["day_energy"] = day_energy
        out["cross_energy"] = cross_energy
        out["energy_denominator"] = np.sqrt(gap_energy * day_energy).replace(0.0, np.nan)
        out["coherence"] = (cross_energy / out["energy_denominator"]).clip(-1.0, 1.0)
        out["gap_drift"] = out["gap"].rolling(params.drift_window).sum()
        out["day_drift"] = out["intraday"].rolling(params.drift_window).sum()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out["coherence"].ge(params.coherence_threshold)
        bullish = out[["gap_drift", "day_drift", "close_return"]].gt(0).all(axis=1)
        bearish = out[["gap_drift", "day_drift", "close_return"]].lt(0).all(axis=1)
        out["candidate"] = np.select([ready & bullish, ready & bearish], [1, -1], default=0)
        out["confirmed"] = out["candidate"].where(
            out["candidate"].eq(out["candidate"].shift(1)), 0
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapEnergyParams,
    ) -> SignalFrame:
        if params.max_hold < 1 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Holding period and profit target must be positive")
        confirmed = indicators["confirmed"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    # Previous close's instruction fills at this open.
                    # ponytail: raw-open fill proxy; use simulator fills if exposed by the contract.
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(closes[i]):
                    position = 0
                else:
                    gain = position * (closes[i] / entry_price - 1.0)
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                        position = 0
            elif confirmed[i] and i + 1 < len(data):
                position = int(confirmed[i])
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=64, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 108.0, "low": 92.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    strategy = GeneratedStrategy()
    params = GapEnergyParams(max_hold=3)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    calculated = strategy.indicators(data, params)
    assert calculated["confirmed"].eq(0).all()
    assert strategy.warmup_bars(params) >= 8 + params.energy_window
    for direction in (1, -1):
        forced = pd.DataFrame({"confirmed": [0, direction] + [0] * 62}, index=index)
        timed = strategy.generate_signals(data, forced, ctx, params).data
        assert timed["signal"].iloc[:6].tolist() == [0, 0, direction, direction, direction, 0]
        target_data = data.copy()
        target_data.iloc[2, target_data.columns.get_loc("close")] = 100.0 * (1.0 + direction * 0.07)
        targeted = strategy.generate_signals(target_data, forced, ctx, params).data
        assert targeted["signal"].iloc[:4].tolist() == [0, 0, direction, 0]
        assert targeted["size"].gt(0).all()
    wave_data = data.copy()
    gap = 0.001 + 0.0005 * np.sin(np.arange(len(data)))
    cumulative = np.cumsum(gap)
    wave_data["open"] = 100.0 * np.exp(2.0 * cumulative - gap)
    wave_data["close"] = 100.0 * np.exp(2.0 * cumulative)
    wave_data["high"] = wave_data[["open", "close"]].max(axis=1) * 1.01
    wave_data["low"] = wave_data[["open", "close"]].min(axis=1) * 0.99
    calculated = strategy.indicators(wave_data, params)
    assert calculated["candidate"].eq(1).any()
    first = int(np.flatnonzero(calculated["candidate"].to_numpy() == 1)[0])
    assert calculated["confirmed"].iloc[first] == 0
    assert calculated["confirmed"].iloc[first + 1] == 1


if __name__ == "__main__":
    _self_check()
