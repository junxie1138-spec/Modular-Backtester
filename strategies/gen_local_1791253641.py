from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    lookback: int = 6
    vol_window: int = 20
    snr_threshold: float = 0.8
    migration_threshold: float = 0.25
    target_vol: float = 0.10
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791253641"

    @classmethod
    def params_type(cls):
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return int(max(params.lookback, params.vol_window) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.lookback < 2 or params.vol_window < 2:
            raise ValueError("Lookback windows must be at least two bars.")
        if not 3 <= params.holding_bars <= 5:
            raise ValueError("Holding period must be three to five bars.")
        if not np.isfinite(params.target_vol) or params.target_vol <= 0:
            raise ValueError("Target volatility must be finite and positive.")
        if not np.isfinite(params.snr_threshold) or params.snr_threshold <= 0:
            raise ValueError("Signal threshold must be finite and positive.")
        if not np.isfinite(params.migration_threshold) or not 0 < params.migration_threshold <= 1:
            raise ValueError("Migration threshold must lie in (0, 1].")

        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_sum"] = out["return"].rolling(params.lookback).sum()
        out["return_energy"] = out["return"].pow(2).rolling(params.lookback).sum()
        out["snr"] = out["return_sum"] / np.sqrt(out["return_energy"].replace(0.0, np.nan))
        out["upper_extension"] = data["high"].diff().clip(lower=0.0)
        out["lower_extension"] = (-data["low"].diff()).clip(lower=0.0)
        out["upper_total"] = out["upper_extension"].rolling(params.lookback).sum()
        out["lower_total"] = out["lower_extension"].rolling(params.lookback).sum()
        out["migration"] = (out["upper_total"] - out["lower_total"]) / (out["upper_total"] + out["lower_total"]).replace(0.0, np.nan)
        out["annual_vol"] = out["return"].rolling(params.vol_window).std(ddof=0) * np.sqrt(252.0)
        out["entry_size"] = (params.target_vol / out["annual_vol"].replace(0.0, np.nan)).clip(lower=0.05, upper=0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        ready = pd.Series(
            np.isfinite(indicators[["snr", "migration", "entry_size"]].to_numpy()).all(axis=1),
            index=data.index,
        )
        long_entry = ready & (indicators["snr"] >= params.snr_threshold) & (indicators["migration"] >= params.migration_threshold)
        short_entry = ready & (indicators["snr"] <= -params.snr_threshold) & (indicators["migration"] <= -params.migration_threshold)
        candidate = np.select([long_entry, short_entry], [1, -1], default=0)
        entry_sizes = indicators["entry_size"].to_numpy()
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.95, dtype=float)
        position, exit_bar, held_size = 0, -1, 0.95

        for i in range(len(data)):
            if position and i == exit_bar:
                position = 0
            elif position == 0 and candidate[i] != 0:
                position = int(candidate[i])
                exit_bar = i + params.holding_bars
                held_size = float(entry_sizes[i])
            signals[i] = position
            if position:
                sizes[i] = held_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    direction = np.where(np.arange(12) < 5, 1.0, -1.0)
    sizes = np.linspace(0.2, 0.8, 12)
    indicators = pd.DataFrame({"snr": direction, "migration": direction, "entry_size": sizes}, index=index)
    params = StrategyParams()
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, -1]
    assert result["size"].iloc[1:5].eq(sizes[0]).all()
    assert result["size"].iloc[6:10].eq(sizes[5]).all()
    indicators["migration"] = -direction
    assert GeneratedStrategy.generate_signals(data, indicators, None, params).data["signal"].eq(0).all()
    flat = GeneratedStrategy.indicators(data, params)
    assert GeneratedStrategy.generate_signals(data, flat, None, params).data["signal"].eq(0).all()
    assert GeneratedStrategy.warmup_bars(params) == 21


if __name__ == "__main__":
    _self_check()
