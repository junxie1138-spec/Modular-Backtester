from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    history: int = 252
    min_analogs: int = 4
    min_agreement: float = 0.65
    edge_atr: float = 0.15
    holding_days: int = 4
    atr_period: int = 14
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790852893"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(200 + params.history + params.holding_days + 5, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if params.history < 1 or params.atr_period < 1 or params.min_analogs < 1:
            raise ValueError("Lookbacks and minimum analog count must be positive")
        if not 3 <= params.holding_days <= 5:
            raise ValueError("holding_days must be between 3 and 5")
        if not 0.5 <= params.min_agreement <= 1 or params.edge_atr < 0 or params.stop_atr <= 0:
            raise ValueError("Invalid agreement, forecast threshold, or stop multiplier")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["acceleration"] = out["return"].diff()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["regime"] = pd.Series(np.where(close > out["ma200"], 1, np.where(close < out["ma200"], -1, 0)), index=data.index).where(out["ma200"].notna() & close.notna())
        previous = close.shift(1)
        out["true_range"] = pd.concat([data["high"] - data["low"], (data["high"] - previous).abs(), (data["low"] - previous).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        word = (out["return"] > 0).astype(int) * 16
        for lag in range(4):
            word = word + (out["acceleration"].shift(lag) > 0).astype(int) * (2 ** lag)
        valid_shape = out["return"].rolling(5, min_periods=5).count().eq(5)
        out["shape"] = word.where(valid_shape)
        offset = params.holding_days + 5
        out["sample_shape"] = out["shape"].shift(offset)
        out["sample_regime"] = out["regime"].shift(offset)
        out["matured_return"] = (close.shift(5) / close.shift(offset) - 1).replace([np.inf, -np.inf], np.nan)
        out["analog_count"] = np.nan
        out["forecast"] = np.nan
        out["agreement"] = np.nan
        for regime in (-1, 1):
            for code in range(32):
                sample = out["sample_shape"].eq(code) & out["sample_regime"].eq(regime) & out["matured_return"].notna()
                count = sample.astype(float).rolling(params.history, min_periods=params.history).sum()
                total = out["matured_return"].where(sample, 0.0).rolling(params.history, min_periods=params.history).sum()
                wins = (sample & (regime * out["matured_return"] > 0)).astype(float).rolling(params.history, min_periods=params.history).sum()
                denominator = count.replace(0, np.nan)
                current = out["shape"].eq(code) & out["regime"].eq(regime)
                out.loc[current, "analog_count"] = count.loc[current]
                out.loc[current, "forecast"] = (total / denominator).loc[current]
                out.loc[current, "agreement"] = (wins / denominator).loc[current]
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["regime"].to_numpy(dtype=float)
        forecast = indicators["forecast"].to_numpy(dtype=float)
        count = indicators["analog_count"].to_numpy(dtype=float)
        agreement = indicators["agreement"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_atr = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position:
                # The prior close's decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                else:
                    stopped = np.isfinite(close[i]) and position * (close[i] - entry_price) < -params.stop_atr * entry_atr
                    expired = i - entry_bar >= params.holding_days - 1
                    if stopped or expired:
                        position = 0
            elif i + 1 < len(data):
                finite = np.isfinite([close[i], atr[i], regime[i], forecast[i], count[i], agreement[i]]).all()
                if finite and close[i] > 0 and atr[i] > 0:
                    direction = int(regime[i])
                    if direction and count[i] >= params.min_analogs and agreement[i] >= params.min_agreement and direction * forecast[i] > params.edge_atr * atr[i] / close[i]:
                        position = direction
                        entry_atr = atr[i]
                        entry_bar = i + 1
                        entry_price = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = AnalogParams()
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    features = pd.DataFrame({"atr": 2.0, "regime": 1.0, "forecast": 0.02, "analog_count": 10.0, "agreement": 1.0}, index=index)
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].iloc[:6].tolist() == [0, 1, 1, 1, 1, 0]
    for direction in (1, -1):
        sample = data.copy()
        sample.loc[index[2], "close"] = 100.0 - direction * 5.0
        features["regime"] = float(direction)
        features["forecast"] = direction * 0.02
        result = strategy.generate_signals(sample, features, None, params).data
        assert result["signal"].iloc[:4].tolist() == [0, direction, direction, 0]
        assert result["size"].gt(0).all()
