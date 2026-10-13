from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalParams:
    history_bars: int = 756
    hold_bars: int = 18
    atr_bars: int = 14
    min_samples: int = 20
    entry_edge: float = 0.10
    sizing_scale: float = 0.30
    stop_mult: float = 2.5


class GeneratedStrategy(BaseStrategy[SeasonalParams]):
    strategy_id = "gen_local_1791248089"

    @classmethod
    def params_type(cls):
        return SeasonalParams

    def warmup_bars(self, params):
        return int(max(params.history_bars + params.hold_bars + 1, params.atr_bars + 1))

    def indicators(self, data, params):
        if (params.history_bars < 2 or params.hold_bars < 1
                or params.atr_bars < 1 or params.min_samples < 2
                or params.entry_edge < 0 or params.sizing_scale <= 0
                or params.stop_mult <= 0):
            raise ValueError("Invalid seasonal strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["return"] = close.pct_change(fill_method=None)
        out["phase"] = (data.index.month - 1) * 2 + (data.index.day > 15).astype(int)
        out["origin_phase"] = out["phase"].shift(params.hold_bars)
        out["matured_return"] = close.pct_change(params.hold_bars, fill_method=None)
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["baseline"] = out["matured_return"].rolling(params.history_bars).mean()
        out["seasonal_mean"] = np.nan
        out["seasonal_std"] = np.nan
        out["seasonal_count"] = 0.0
        # Outcomes are assigned to their starting calendar phase only after maturity.
        for phase in range(24):
            samples = out["matured_return"].where(out["origin_phase"].eq(phase))
            window = samples.rolling(params.history_bars, min_periods=params.min_samples)
            select = out["phase"].eq(phase)
            out.loc[select, "seasonal_mean"] = window.mean().loc[select]
            out.loc[select, "seasonal_std"] = window.std(ddof=0).loc[select]
            out.loc[select, "seasonal_count"] = samples.rolling(
                params.history_bars, min_periods=1
            ).count().loc[select]
        denominator = out["seasonal_std"].where(out["seasonal_std"] > 1e-12)
        out["edge"] = ((out["seasonal_mean"] - out["baseline"]) / denominator).replace(
            [np.inf, -np.inf], np.nan
        )
        out["edge"] = out["edge"].where(
            np.arange(len(out)) >= self.warmup_bars(params) - 1
        )
        return out

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        edge = indicators["edge"].to_numpy(dtype=float)
        phases = indicators["phase"].to_numpy(dtype=int)
        active = False
        stop = np.nan
        entry_atr = np.nan
        held = 0
        allocation = 1.0
        for i in range(n):
            new_phase = i == 0 or phases[i] != phases[i - 1]
            if active:
                if held == 0:
                    # This open is observed on the fill bar, never on the decision bar.
                    stop = opens[i] - params.stop_mult * entry_atr
                held += 1
                if (not np.isfinite(closes[i]) or not np.isfinite(stop)
                        or closes[i] < stop or held >= params.hold_bars):
                    active = False
            elif (new_phase and np.isfinite(edge[i])
                  and edge[i] > params.entry_edge
                  and np.isfinite(atr[i]) and atr[i] > 0
                  and np.isfinite(closes[i]) and closes[i] > 0):
                active = True
                entry_atr = atr[i]
                held = 0
                allocation = float(np.clip(
                    0.20 + 0.75 * (edge[i] - params.entry_edge) / params.sizing_scale,
                    0.20, 0.95
                ))
            signals[i] = int(active)
            sizes[i] = allocation if active else 1.0
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = SeasonalParams()
    index = pd.bdate_range("2020-01-02", periods=30)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0
    }, index=index)
    indicators = pd.DataFrame({"atr": 2.0, "edge": 0.0, "phase": 0}, index=index)
    indicators.iloc[0, indicators.columns.get_loc("edge")] = 0.20
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["signal"].iloc[19:].eq(0).all()
    assert result["size"].gt(0).all()
    stronger = indicators.copy()
    stronger.iloc[0, stronger.columns.get_loc("edge")] = 0.30
    assert strategy.generate_signals(data, stronger, ctx, params).data["size"].iloc[1] > result["size"].iloc[1]
    stopped = data.copy()
    stopped.iloc[2, stopped.columns.get_loc("close")] = 110.0
    stopped.iloc[3, stopped.columns.get_loc("close")] = 99.0
    stopped.iloc[4, stopped.columns.get_loc("close")] = 94.0
    result = strategy.generate_signals(stopped, indicators, ctx, params).data
    assert result["signal"].iloc[4] == 1
    assert result["signal"].iloc[5] == 0
    gapped = data.copy()
    gapped.iloc[1, gapped.columns.get_loc("open")] = 110.0
    gapped.iloc[1, gapped.columns.get_loc("close")] = 110.0
    gapped.iloc[2, gapped.columns.get_loc("close")] = 104.0
    assert strategy.generate_signals(gapped, indicators, ctx, params).data["signal"].iloc[3] == 0


if __name__ == "__main__":
    _self_check()
