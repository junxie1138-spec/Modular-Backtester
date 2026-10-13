from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    lookback: int = 6
    area_threshold: float = 0.30
    efficiency_threshold: float = 0.20
    reset_area: float = 0.0
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1790895200"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        if not 2 <= params.lookback <= 9:
            raise ValueError("lookback must be between 2 and 9")
        if not -1 <= params.reset_area < params.area_threshold <= 1:
            raise ValueError("invalid signature thresholds")
        if not 0 < params.efficiency_threshold <= 1:
            raise ValueError("efficiency_threshold must be in (0, 1]")
        if params.profit_target <= 0 or params.max_hold_bars < 1:
            raise ValueError("profit target and holding period must be positive")
        return int(params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(self, params)
        n = params.lookback
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        up = returns.clip(lower=0)
        down = (-returns).clip(lower=0)
        up_total = up.rolling(n, min_periods=n).sum()
        down_total = down.rolling(n, min_periods=n).sum()
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, n):
            ordered_pairs = down.shift(lag) * up - up.shift(lag) * down
            area = area + ordered_pairs.rolling(n - lag, min_periods=n - lag).sum()
        normalization = (up_total * down_total).replace(0.0, np.nan)
        movement = (up_total + down_total).replace(0.0, np.nan)
        out = pd.DataFrame({
            "upward_step": up,
            "downward_step": down,
            "upward_variation": up_total,
            "downward_variation": down_total,
            "signature_area": (area / normalization).clip(-1.0, 1.0),
            "trend_efficiency": (up_total - down_total) / movement,
        }, index=data.index)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(self, params)
        area = indicators["signature_area"].fillna(0.0).to_numpy()
        efficiency = indicators["trend_efficiency"].fillna(0.0).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        held = False
        armed = True
        entry_price = np.nan
        age = 0
        for i in range(len(data)):
            if area[i] <= params.reset_area:
                armed = True
            if held:
                if np.isnan(entry_price):
                    entry_price = opens[i]
                age += 1
                target_hit = (
                    np.isfinite(entry_price)
                    and entry_price > 0
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if target_hit or age >= params.max_hold_bars:
                    held = False
                    entry_price = np.nan
                    age = 0
                else:
                    signal[i] = 1
                continue
            if (
                armed
                and np.isfinite(closes[i])
                and closes[i] > 0
                and area[i] >= params.area_threshold
                and efficiency[i] >= params.efficiency_threshold
            ):
                held = True
                armed = False
                entry_price = np.nan
                age = 0
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = SignatureParams()
    index = pd.date_range("2020-01-01", periods=14)
    data = pd.DataFrame({
        "open": 100.0, "high": 100.0, "low": 100.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    features = pd.DataFrame({"signature_area": 0.8, "trend_efficiency": 0.6}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, features, None, params).data
    assert result["signal"].tolist() == [0] + [1] * 8 + [0] * 5
    data.loc[index[3], ["high", "close"]] = 103.0
    result = GeneratedStrategy.generate_signals(None, data, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1] + [0] * 10
    derived = GeneratedStrategy.indicators(None, data, params)
    assert derived.index.equals(data.index)
    assert np.isfinite(derived.to_numpy()).all()
    assert GeneratedStrategy.warmup_bars(None, params) <= 10
