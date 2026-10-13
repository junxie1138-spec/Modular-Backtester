from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureSeasonParams:
    signature_bars: int = 8
    area_threshold: float = 0.12
    pressure_bars: int = 3
    pressure_threshold: float = 0.15
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[SignatureSeasonParams]):
    strategy_id = "gen_local-w2_1791242281"

    @classmethod
    def params_type(cls):
        return SignatureSeasonParams

    def warmup_bars(self, params: SignatureSeasonParams) -> int:
        return max(params.signature_bars + 1, params.pressure_bars)

    def indicators(self, data: pd.DataFrame, params: SignatureSeasonParams) -> pd.DataFrame:
        if params.signature_bars < 2 or params.pressure_bars < 1 or params.hold_bars < 1:
            raise ValueError("Lookbacks and holding period must be positive; signature_bars must exceed one.")
        if not 0.0 < params.area_threshold < 0.5 or not 0.0 <= params.pressure_threshold < 1.0:
            raise ValueError("Invalid area or closing-pressure threshold.")
        n = params.signature_bars
        change = data["close"].diff()
        volume = data["volume"].clip(lower=0).fillna(0.0)
        buy = volume.where(change > 0, 0.0)
        sell = volume.where(change < 0, 0.0)
        area = pd.Series(0.0, index=data.index)
        # Sum antisymmetric level-two signature terms over ordered pairs.
        for lag in range(1, n):
            cross = buy.shift(lag) * sell - sell.shift(lag) * buy
            area = area + 0.5 * cross.rolling(n - lag, min_periods=n - lag).sum()
        scale = buy.rolling(n, min_periods=n).sum() * sell.rolling(n, min_periods=n).sum()
        area = (area / scale.where(scale > 0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        spread = (data["high"] - data["low"]).where(data["high"] > data["low"])
        location = ((2.0 * data["close"] - data["high"] - data["low"]) / spread).clip(-1.0, 1.0).fillna(0.0)
        m = params.pressure_bars
        pressure = (volume * location).rolling(m, min_periods=m).sum()
        pressure = pressure / volume.rolling(m, min_periods=m).sum().replace(0.0, np.nan)
        pressure = pressure.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        calendar = pd.Series(
            (data.index.day >= data.index.days_in_month - 5) | (data.index.day <= 3),
            index=data.index,
        )
        valid = pd.Series(np.isfinite(data[["open", "high", "low", "close", "volume"]]).all(axis=1), index=data.index)
        valid = valid & (data["close"] > 0) & (data["volume"] >= 0) & (data["high"] >= data["low"])
        window = self.warmup_bars(params)
        ready = valid.rolling(window, min_periods=window).sum().eq(window)
        return pd.DataFrame({
            "volume_signature_area": area,
            "closing_pressure": pressure,
            "seasonal_window": calendar,
            "ready": ready,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureSeasonParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive.")
        entry = (
            indicators["ready"]
            & indicators["seasonal_window"]
            & (indicators["volume_signature_area"] <= -params.area_threshold)
            & (indicators["closing_pressure"] > params.pressure_threshold)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
                    # Keep an explicit flat bar; no same-bar re-entry.
            elif entry[i]:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = SignatureSeasonParams()
    index = pd.date_range("2020-01-01", periods=60, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    computed = strategy.indicators(data, params)
    assert computed.notna().all().all()
    assert not computed["ready"].iloc[:strategy.warmup_bars(params) - 1].any()
    forced = pd.DataFrame({"ready": True, "seasonal_window": True,
                           "volume_signature_area": -0.3, "closing_pressure": 0.5}, index=index)
    result = strategy.generate_signals(data, forced, None, params).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["signal"].iloc[19] == 0
    assert result["signal"].iloc[20:38].eq(1).all()
    forced["closing_pressure"] = 0.0
    assert strategy.generate_signals(data, forced, None, params).data["signal"].eq(0).all()
    forced["closing_pressure"] = 0.5
    forced["volume_signature_area"] = 0.0
    assert strategy.generate_signals(data, forced, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
