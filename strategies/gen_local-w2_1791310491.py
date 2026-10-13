from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ChangepointParams:
    ma_window: int = 40
    recent_window: int = 8
    reference_window: int = 20
    z_threshold: float = 0.65
    pressure_threshold: float = 0.12


class GeneratedStrategy(BaseStrategy[ChangepointParams]):
    strategy_id = "gen_local-w2_1791310491"

    @classmethod
    def params_type(cls):
        return ChangepointParams

    def warmup_bars(self, params: ChangepointParams) -> int:
        if min(params.ma_window, params.recent_window, params.reference_window) < 2:
            raise ValueError("Lookback windows must be at least two bars.")
        if not np.isfinite(params.z_threshold) or params.z_threshold <= 0:
            raise ValueError("z_threshold must be finite and positive.")
        if not np.isfinite(params.pressure_threshold) or params.pressure_threshold <= 0:
            raise ValueError("pressure_threshold must be finite and positive.")
        return int(params.ma_window + params.recent_window + params.reference_window - 1)

    def indicators(self, data: pd.DataFrame, params: ChangepointParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].replace([np.inf, -np.inf], np.nan)
        out["ma"] = close.rolling(params.ma_window).mean()
        out["price_std"] = close.rolling(params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["price_std"].where(out["price_std"] > 0)
        out["z_recent"] = out["z"].rolling(params.recent_window).mean()
        out["z_reference"] = out["z"].rolling(params.reference_window).mean().shift(params.recent_window)
        out["z_change"] = out["z_recent"] - out["z_reference"]

        out["range"] = (data["high"] - data["low"]).replace([np.inf, -np.inf], np.nan)
        out["body"] = (data["close"] - data["open"]).replace([np.inf, -np.inf], np.nan)
        out["body_fraction"] = (out["body"] / out["range"].where(out["range"] > 0)).clip(-1.0, 1.0)
        zero_range = out["range"].eq(0) & out["body"].eq(0)
        out.loc[zero_range, "body_fraction"] = 0.0
        out["valid_volume"] = data["volume"].replace([np.inf, -np.inf], np.nan).where(data["volume"] >= 0)
        out["weighted_body"] = out["body_fraction"] * out["valid_volume"]
        out["volume_recent"] = out["valid_volume"].rolling(params.recent_window).sum()
        out["volume_reference"] = out["valid_volume"].rolling(params.reference_window).sum().shift(params.recent_window)
        out["body_recent"] = out["weighted_body"].rolling(params.recent_window).sum()
        out["body_reference"] = out["weighted_body"].rolling(params.reference_window).sum().shift(params.recent_window)
        out["pressure_recent"] = out["body_recent"] / out["volume_recent"].where(out["volume_recent"] > 0)
        out["pressure_reference"] = out["body_reference"] / out["volume_reference"].where(out["volume_reference"] > 0)
        out["pressure_change"] = out["pressure_recent"] - out["pressure_reference"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ChangepointParams) -> SignalFrame:
        price_up = (indicators["z_change"] >= params.z_threshold) & (indicators["z_recent"] > 0)
        price_down = (indicators["z_change"] <= -params.z_threshold) & (indicators["z_recent"] < 0)
        pressure_up = (indicators["pressure_change"] >= params.pressure_threshold) & (indicators["pressure_recent"] > 0)
        pressure_down = (indicators["pressure_change"] <= -params.pressure_threshold) & (indicators["pressure_recent"] < 0)
        proposal = np.select([(price_up & pressure_up).fillna(False), (price_down & pressure_down).fillna(False)], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        state = 0
        for i, direction in enumerate(proposal):
            if state == 0:
                state = int(direction)
            elif direction == -state:
                # An opposite joint condition exits; subsequent agreement can reenter.
                state = 0
            positions[i] = state
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = ChangepointParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    direction = pd.Series([1.0, 1.0, -1.0, -1.0, np.nan, 1.0, 1.0], index=index)
    derived = pd.DataFrame({"z_change": direction, "z_recent": direction, "pressure_change": direction, "pressure_recent": direction}, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    assert strategy.indicators(data, params)["z_change"].isna().all()
    assert strategy.warmup_bars(params) == 67
