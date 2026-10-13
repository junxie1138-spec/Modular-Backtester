from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    fractional_order: float = 0.45
    fractional_window: int = 48
    coherence_window: int = 10
    percentile_window: int = 126
    entry_quantile: float = 0.85
    volatility_window: int = 20
    target_annual_vol: float = 0.12
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791323152"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(
            params.fractional_window + params.coherence_window + params.percentile_window + 2,
            params.volatility_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        periods = (params.fractional_window, params.coherence_window,
                   params.percentile_window, params.volatility_window)
        if (any(int(p) != p or p < 2 for p in periods)
                or int(params.hold_bars) != params.hold_bars or params.hold_bars < 1
                or not 0.0 < params.fractional_order < 1.0
                or not 0.5 < params.entry_quantile < 1.0
                or not np.isfinite(params.target_annual_vol)
                or params.target_annual_vol <= 0.0):
            raise ValueError("Invalid fractional trend or volatility parameters")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0.0))
        fractional = out["log_close"].copy()
        weight = 1.0
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1.0 - params.fractional_order) / lag
            fractional = fractional + weight * out["log_close"].shift(lag)
        out["fractional_close"] = fractional
        out["fractional_change"] = fractional.diff()
        out["drift"] = out["fractional_change"].rolling(params.coherence_window).mean()
        out["noise"] = out["fractional_change"].rolling(params.coherence_window).std()
        out["coherence"] = (out["drift"] / out["noise"].replace(0.0, np.nan)).replace(
            [np.inf, -np.inf], np.nan)
        out["strength"] = out["coherence"].abs()
        out["threshold"] = out["strength"].shift(1).rolling(
            params.percentile_window).quantile(params.entry_quantile)
        out["log_return"] = out["log_close"].diff()
        out["annual_vol"] = out["log_return"].rolling(
            params.volatility_window).std() * np.sqrt(252.0)
        out["risk_size"] = (params.target_annual_vol /
                            out["annual_vol"].replace(0.0, np.nan)).replace(
                                [np.inf, -np.inf], np.nan).clip(0.05, 1.0)
        out["entry"] = (
            (out["strength"] > out["threshold"])
            & (out["strength"].shift(1) <= out["threshold"].shift(1))
            & out["risk_size"].notna()
        )
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        direction = np.sign(indicators["coherence"].fillna(0.0)).to_numpy()
        risk = indicators["risk_size"].to_numpy(dtype=float)
        position, exit_bar, entry_size = 0, -1, 1.0
        for i in range(len(data)):
            if position != 0:
                if i == exit_bar:
                    position = 0
                    continue
            elif entries[i] and direction[i] != 0 and np.isfinite(risk[i]) and risk[i] > 0:
                position = int(direction[i])
                exit_bar = i + params.hold_bars
                entry_size = float(risk[i])
            signal[i] = position
            if position != 0:
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GeneratedParams()
    strategy = GeneratedStrategy()
    count = 2 * params.hold_bars + 6
    index = pd.date_range("2020-01-01", periods=count, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"entry": False, "coherence": 1.0,
                        "risk_size": 0.4}, index=index)
    ind.iloc[0, ind.columns.get_loc("entry")] = True
    second = params.hold_bars + 2
    ind.iloc[second, ind.columns.get_loc("entry")] = True
    ind.iloc[1:, ind.columns.get_loc("coherence")] = -1.0
    ind.iloc[1:, ind.columns.get_loc("risk_size")] = 0.7
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].iloc[0] == 0
    assert (result["signal"].iloc[1:params.hold_bars + 1] == 1).all()
    assert result["signal"].iloc[params.hold_bars + 1] == 0
    assert (result["size"].iloc[1:params.hold_bars + 1] == 0.4).all()
    assert (result["signal"].iloc[second + 1:second + params.hold_bars + 1] == -1).all()
    assert result["signal"].iloc[second + params.hold_bars + 1] == 0
    assert np.isfinite(result["size"]).all() and (result["size"] > 0).all()
    flat = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert (flat["signal"] == 0).all()


if __name__ == "__main__":
    _self_check()
