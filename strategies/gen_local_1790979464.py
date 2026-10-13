from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureRecoveryParams:
    path_window: int = 6
    drawdown_window: int = 20
    volume_window: int = 20
    min_drawdown: float = 0.015
    area_threshold: float = 0.06
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SignatureRecoveryParams]):
    strategy_id = "gen_local_1790979464"

    @classmethod
    def params_type(cls) -> type[SignatureRecoveryParams]:
        return SignatureRecoveryParams

    def warmup_bars(self, params: SignatureRecoveryParams) -> int:
        return int(max(200, params.path_window + 1,
                       params.volume_window + params.path_window,
                       params.drawdown_window + params.path_window))

    def indicators(self, data: pd.DataFrame, params: SignatureRecoveryParams) -> pd.DataFrame:
        if (params.path_window < 2 or params.drawdown_window < 2
                or params.volume_window < 1 or params.hold_bars not in (1, 2)
                or not np.isfinite(params.min_drawdown)
                or not np.isfinite(params.area_threshold)
                or params.min_drawdown <= 0 or params.area_threshold <= 0):
            raise ValueError("Invalid signature recovery parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["regime_ma"] = close.rolling(200, min_periods=200).mean()
        out["price_increment"] = close.pct_change(fill_method=None)
        out["average_volume"] = data["volume"].rolling(
            params.volume_window, min_periods=params.volume_window).mean()
        candle_range = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["candle_pressure"] = ((close - data["open"]) / candle_range).clip(-1.0, 1.0)
        out.loc[(data["high"] == data["low"]) & close.notna(), "candle_pressure"] = 0.0
        out["pressure_increment"] = (
            out["candle_pressure"] * data["volume"]
            / out["average_volume"].where(out["average_volume"] > 0))
        x = out["price_increment"]
        y = out["pressure_increment"]
        area = pd.Series(0.0, index=data.index)
        # Each pair of increments contributes once to the second-order signature.
        for lag in range(1, params.path_window):
            pairs = x.shift(lag) * y - y.shift(lag) * x
            width = params.path_window - lag
            area = area + pairs.rolling(width, min_periods=width).sum()
        out["signature_area"] = 0.5 * area
        out["price_path_length"] = x.abs().rolling(
            params.path_window, min_periods=params.path_window).sum()
        out["pressure_path_length"] = y.abs().rolling(
            params.path_window, min_periods=params.path_window).sum()
        denominator = out["price_path_length"] * out["pressure_path_length"]
        out["normalized_area"] = out["signature_area"] / denominator.where(denominator > 0)
        out["rolling_peak"] = close.rolling(
            params.drawdown_window, min_periods=params.drawdown_window).max()
        out["drawdown"] = close / out["rolling_peak"] - 1.0
        out["recent_drawdown"] = out["drawdown"].rolling(
            params.path_window, min_periods=params.path_window).min()
        out["previous_return"] = x.shift(1)
        ready_columns = ["regime_ma", "normalized_area", "drawdown",
                         "recent_drawdown", "price_increment", "previous_return"]
        out["ready"] = pd.Series(
            np.isfinite(out[ready_columns].to_numpy()).all(axis=1), index=data.index)
        in_drawdown = ((out["recent_drawdown"] <= -params.min_drawdown)
                       & (out["drawdown"] < -0.25 * params.min_drawdown))
        out["long_setup"] = (
            out["ready"] & in_drawdown & (close > out["regime_ma"])
            & (x > 0) & (out["normalized_area"] < -params.area_threshold))
        out["short_setup"] = (
            out["ready"] & in_drawdown & (close < out["regime_ma"])
            & (out["previous_return"] > 0) & (x < 0)
            & (out["normalized_area"] > params.area_threshold))
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureRecoveryParams) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two sessions")
        longs = indicators["long_setup"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_setup"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=np.int64)
        position = 0
        entry_bar = -1
        for bar in range(len(data)):
            if position:
                if bar - entry_bar >= params.hold_bars:
                    position = 0
            elif longs[bar] or shorts[bar]:
                position = 1 if longs[bar] else -1
                entry_bar = bar
            signals[bar] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "long_setup": [True, False, False, False, False, False, False, False],
        "short_setup": [False, True, False, True, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, SignatureRecoveryParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 0]
    assert (result.data["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
