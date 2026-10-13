from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy

@dataclass(slots=True)
class Params:
    peak_window: int = 40
    flow_window: int = 12
    min_drawdown: float = 0.03
    volume_threshold: float = 1.3
    downside_share: float = 0.60
    hold_bars: int = 18

class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1790973773"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.peak_window + 1, params.flow_window + 2))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.peak_window < 2 or params.flow_window < 2 or params.hold_bars < 1:
            raise ValueError("Lookbacks must be at least two and hold_bars positive.")
        if not (0 < params.min_drawdown < 1 and params.volume_threshold > 0 and 0 < params.downside_share < 1):
            raise ValueError("Invalid drawdown, volume, or downside-share threshold.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        volume = data["volume"].clip(lower=0)
        out["return"] = close.pct_change(fill_method=None)
        out["prior_peak"] = close.shift(1).rolling(params.peak_window).max()
        out["drawdown"] = 1.0 - close / out["prior_peak"].replace(0, np.nan)
        out["selling_volume"] = volume.where(out["return"] < 0, 0.0)
        out["seller_notional"] = out["selling_volume"] * close
        out["prior_selling_volume"] = out["selling_volume"].shift(1).rolling(params.flow_window).sum()
        out["prior_volume"] = volume.shift(1).rolling(params.flow_window).sum()
        out["seller_basis"] = out["seller_notional"].shift(1).rolling(params.flow_window).sum() / out["prior_selling_volume"].replace(0, np.nan)
        out["selling_share"] = out["prior_selling_volume"] / out["prior_volume"].replace(0, np.nan)
        out["relative_volume"] = volume / (out["prior_volume"] / params.flow_window).replace(0, np.nan)
        out["mean_abs_return"] = out["return"].abs().shift(1).rolling(params.flow_window).mean()
        out["move_strength"] = out["return"].abs() / out["mean_abs_return"].replace(0, np.nan)
        out["body_efficiency"] = (close - data["open"]).abs() / (data["high"] - data["low"]).replace(0, np.nan)
        ready = out["mean_abs_return"].notna() & out["seller_basis"].notna() & out["prior_peak"].notna()
        active = ready & (out["drawdown"] >= params.min_drawdown) & (out["selling_share"] >= params.downside_share) & (out["relative_volume"] >= params.volume_threshold) & (out["body_efficiency"] >= 0.35)
        reclaimed = (close.shift(1) <= out["seller_basis"]) & (close > out["seller_basis"]) & (out["return"] > 0) & (close > data["open"])
        failed = (close.shift(1) >= out["seller_basis"]) & (close < out["seller_basis"]) & (out["return"] < 0) & (close < data["open"])
        out["candidate"] = np.select([active & reclaimed, active & failed], [1, -1], default=0).astype(int)
        out["conviction"] = (0.35 + 0.30 * (out["relative_volume"] - 1).clip(0, 1) + 0.30 * (out["move_strength"] / 2).clip(0, 1)).clip(0.35, 0.95)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive.")
        candidate = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        conviction = indicators["conviction"].replace([np.inf, -np.inf], np.nan).fillna(0.35).clip(0.35, 0.95).to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.35, dtype=float)
        position, expiry, entry_size = 0, -1, 0.35
        for i in range(len(data)):
            if position and i == expiry:
                position = 0
                continue
            if position == 0 and candidate[i] != 0:
                position = int(candidate[i])
                expiry = i + params.hold_bars
                entry_size = float(conviction[i])
            if position:
                signal[i] = position
                size[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.35).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")

if __name__ == "__main__":
    p = Params()
    s = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=60)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = s.indicators(data, p)
    assert np.isfinite(derived.to_numpy()).all()
    assert derived["candidate"].eq(0).all()
    probe = pd.DataFrame({"candidate": -1, "conviction": np.linspace(0.4, 0.9, len(index))}, index=index)
    probe.iloc[0, 0] = 1
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=s.warmup_bars(p))
    result = s.generate_signals(data, probe, ctx, p).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["signal"].iloc[19] == 0
    assert result["signal"].iloc[20:38].eq(-1).all()
    assert result["signal"].iloc[38] == 0
    assert result["size"].iloc[1:19].eq(0.4).all()
    assert result["size"].gt(0).all()
