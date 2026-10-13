from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    signature_window: int = 8
    area_threshold: float = 0.15
    holding_bars: int = 18
    target_vol: float = 0.12
    volatility_floor: float = 0.04
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1790916326"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        if not 2 <= params.signature_window <= 9:
            raise ValueError("signature_window must be between 2 and 9")
        return int(params.signature_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        window = params.signature_window
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] > 0)
        dx = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        dy = np.log(volume).diff().replace([np.inf, -np.inf], np.nan)
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, window):
            cross = dx.shift(lag) * dy - dy.shift(lag) * dx
            area = area + 0.5 * cross.rolling(window - lag, min_periods=window - lag).sum()
        scale = np.sqrt(
            dx.pow(2).rolling(window, min_periods=window).sum()
            * dy.pow(2).rolling(window, min_periods=window).sum()
        )
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out["signature_area"] = (area / scale.where(scale > 0)).replace([np.inf, -np.inf], np.nan)
        out["annualized_vol"] = returns.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252.0)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        if params.holding_bars < 1 or params.area_threshold <= 0:
            raise ValueError("holding_bars and area_threshold must be positive")
        if not (params.target_vol > 0 and params.volatility_floor > 0 and 0 < params.max_size <= 1):
            raise ValueError("Invalid volatility sizing parameters")
        area = indicators["signature_area"].to_numpy(dtype=float)
        vol = indicators["annualized_vol"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), float(params.max_size), dtype=float)
        direction = 0
        entry_bar = -1
        entry_size = float(params.max_size)
        for bar in range(len(data)):
            if direction != 0:
                if bar - entry_bar == params.holding_bars:
                    direction = 0
                else:
                    signals[bar] = direction
                    sizes[bar] = entry_size
                continue
            if np.isfinite(area[bar]) and np.isfinite(vol[bar]) and abs(area[bar]) >= params.area_threshold:
                direction = 1 if area[bar] > 0 else -1
                entry_bar = bar
                entry_size = float(min(params.max_size, params.target_vol / max(vol[bar], params.volatility_floor)))
                signals[bar] = direction
                sizes[bar] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["size"] = df["size"].shift(1).fillna(float(params.max_size)).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = SignatureParams()
    index = pd.date_range("2020-01-01", periods=40)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = GeneratedStrategy.indicators(data, params)
    assert derived.index.equals(index)
    assert GeneratedStrategy.warmup_bars(params) <= 10
    for direction in (1, -1):
        supplied = pd.DataFrame({"signature_area": float(direction), "annualized_vol": 0.20}, index=index)
        supplied.iloc[5:, 0] = -float(direction)
        result = GeneratedStrategy.generate_signals(data, supplied, None, params).data
        assert result["signal"].tolist() == [0] + [direction] * 18 + [0] + [-direction] * 18 + [0, -direction]
        assert np.allclose(result["size"].iloc[1:19], 0.60)
        assert (result["size"] > 0).all()
    flat = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert (flat["signal"] == 0).all()
