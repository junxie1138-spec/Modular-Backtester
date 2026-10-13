from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 6
    volume_bars: int = 20
    percentile_bars: int = 126
    entry_quantile: float = 0.80
    min_efficiency: float = 0.55
    profit_target: float = 0.015
    max_hold_bars: int = 4

    def __post_init__(self):
        if min(self.path_bars, self.volume_bars, self.percentile_bars) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not 0.0 < self.entry_quantile < 1.0:
            raise ValueError("entry_quantile must be between zero and one.")
        if not 0.0 <= self.min_efficiency <= 1.0:
            raise ValueError("min_efficiency must be between zero and one.")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0.0:
            raise ValueError("profit_target must be finite and positive.")
        if not 3 <= self.max_hold_bars <= 5:
            raise ValueError("max_hold_bars must be between three and five.")


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791287455"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params):
        return int(params.volume_bars + params.path_bars + params.percentile_bars + 1)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        volume = data["volume"].where(data["volume"] >= 0.0)
        out["log_return"] = np.log(close).diff()
        out["volume_base"] = volume.rolling(params.volume_bars).mean().shift(1)
        out["volume_impulse"] = volume.div(out["volume_base"].where(out["volume_base"] > 0.0)) - 1.0
        out = out.replace([np.inf, -np.inf], np.nan)
        r = out["log_return"]
        v = out["volume_impulse"]
        area = pd.Series(0.0, index=data.index)
        # Half the antisymmetric second-order signature of cumulative (v, r).
        for lag in range(1, params.path_bars):
            cross = v.shift(lag) * r - r.shift(lag) * v
            area = area + 0.5 * cross.rolling(params.path_bars - lag).sum()
        out["raw_area"] = area
        out["volume_travel"] = v.abs().rolling(params.path_bars).sum()
        out["return_travel"] = r.abs().rolling(params.path_bars).sum()
        out["net_return"] = r.rolling(params.path_bars).sum()
        scale = out["volume_travel"] * out["return_travel"]
        out["signature_area"] = area.div(scale.where(scale > 0.0))
        out["efficiency"] = out["net_return"].div(out["return_travel"].where(out["return_travel"] > 0.0))
        out["area_threshold"] = out["signature_area"].shift(1).rolling(params.percentile_bars).quantile(params.entry_quantile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        entry = (
            (indicators["signature_area"] > indicators["area_threshold"])
            & (indicators["signature_area"] > 0.0)
            & (indicators["efficiency"] >= params.min_efficiency)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                # Read the execution open only once that bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        active = False
                        continue
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_up = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or time_up:
                    active = False
                else:
                    signal[i] = 1
            elif entry[i]:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = SignatureParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": [50.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
         "high": 103.0, "low": 49.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    features = pd.DataFrame(
        {"signature_area": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         "area_threshold": 0.1, "efficiency": 0.8}, index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    data.loc[index[2], "close"] = 102.0
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    features.loc[index[0], "area_threshold"] = np.nan
    assert strategy.generate_signals(data, features, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
