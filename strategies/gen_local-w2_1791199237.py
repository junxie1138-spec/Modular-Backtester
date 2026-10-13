from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PlasticSlopeParams:
    slope_window: int = 20
    noise_window: int = 40
    memory_window: int = 10
    yield_threshold: float = 1.25
    balance_threshold: float = 0.35
    min_efficiency: float = 0.20
    profit_target: float = 0.04
    max_holding_bars: int = 18
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[PlasticSlopeParams]):
    strategy_id = "gen_local-w2_1791199237"

    @classmethod
    def params_type(cls):
        return PlasticSlopeParams

    def warmup_bars(self, params: PlasticSlopeParams) -> int:
        return int(params.slope_window + params.noise_window + params.memory_window + 2)

    def indicators(self, data: pd.DataFrame, params: PlasticSlopeParams) -> pd.DataFrame:
        if min(params.slope_window, params.noise_window, params.memory_window) < 2:
            raise ValueError("Indicator windows must be at least two.")
        if params.yield_threshold <= 0 or not 0 < params.balance_threshold < 1:
            raise ValueError("Invalid innovation thresholds.")
        if not 0 <= params.min_efficiency < 1:
            raise ValueError("Invalid efficiency threshold.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["slope"] = out["return"].rolling(params.slope_window).mean()
        out["innovation"] = out["slope"].diff()
        out["innovation_noise"] = out["innovation"].rolling(params.noise_window).std(ddof=0).shift(1)
        out["innovation_z"] = out["innovation"].div(out["innovation_noise"].where(out["innovation_noise"] > 1e-12))
        # Sub-yield revisions are elastic; only excess revisions leave plastic memory.
        out["plastic_revision"] = np.sign(out["innovation_z"]) * (out["innovation_z"].abs() - params.yield_threshold).clip(lower=0)
        out["plastic_net"] = out["plastic_revision"].rolling(params.memory_window).sum()
        out["plastic_total"] = out["plastic_revision"].abs().rolling(params.memory_window).sum()
        out["balance"] = out["plastic_net"].div(out["plastic_total"].where(out["plastic_total"] > 1e-12))
        out["net_return"] = out["return"].rolling(params.slope_window).sum()
        out["gross_return"] = out["return"].abs().rolling(params.slope_window).sum()
        out["efficiency"] = out["net_return"].div(out["gross_return"].where(out["gross_return"] > 1e-12))
        out["strength"] = out["balance"].clip(0, 1) * out["efficiency"].clip(0, 1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: PlasticSlopeParams) -> SignalFrame:
        if params.profit_target <= 0 or params.max_holding_bars < 1:
            raise ValueError("Profit target and holding limit must be positive.")
        if not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError("Position sizes must lie in (0, 1].")
        eligible = (
            (indicators["slope"] > 0)
            & (indicators["balance"] >= params.balance_threshold)
            & (indicators["efficiency"] >= params.min_efficiency)
        ).fillna(False).to_numpy(dtype=bool)
        strength = indicators["strength"].fillna(0).clip(0, 1).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        active = False
        entry_bar = -1
        entry_price = np.nan
        trade_size = params.min_size
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                target_hit = valid_entry and np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                expired = i - entry_bar + 1 >= params.max_holding_bars
                if not valid_entry or target_hit or expired:
                    active = False
                # Exposure stays fixed until exit; no additions reset the trade clock.
            elif i >= warmup and eligible[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                trade_size = params.min_size + (params.max_size - params.min_size) * strength[i]
            signal[i] = int(active)
            sizes[i] = trade_size if active else params.min_size
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = PlasticSlopeParams(slope_window=2, noise_window=2, memory_window=2, max_holding_bars=3)
    strategy = GeneratedStrategy()
    start = strategy.warmup_bars(params)
    index = pd.date_range("2020-01-01", periods=start + 9)
    data = pd.DataFrame({"open": 100.0, "high": 106.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    data.iloc[start + 3, data.columns.get_loc("close")] = 105.0
    ind = pd.DataFrame({"slope": 0.01, "balance": 0.8, "efficiency": 1.0,
                        "strength": 0.8}, index=index)
    ind.iloc[start, ind.columns.get_loc("balance")] = 0.0
    ind.iloc[start + 4, ind.columns.get_loc("strength")] = 0.4
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=start)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].iloc[:start].eq(0).all()
    assert result["signal"].iloc[start:].tolist() == [0, 0, 1, 1, 0, 1, 1, 1, 0]
    assert np.isclose(result["size"].iloc[start + 2], 0.81)
    assert np.isclose(result["size"].iloc[start + 5], 0.53)
    assert result["size"].gt(0).all()
    derived = strategy.indicators(data, params)
    actual = strategy.generate_signals(data, derived, ctx, params).data
    assert actual["signal"].eq(0).all()
    assert np.isfinite(actual["size"]).all()


if __name__ == "__main__":
    _self_check()
