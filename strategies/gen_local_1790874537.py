from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fractional_order: float = 0.45
    fractional_window: int = 32
    z_window: int = 20
    wave_window: int = 3
    vol_window: int = 20
    fd_threshold: float = 0.75
    wave_threshold: float = 0.08
    target_vol: float = 0.12
    profit_target: float = 0.015
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790874537"

    @classmethod
    def params_type(cls):
        return StrategyParams

    @staticmethod
    def warmup_bars(params):
        return int(max(params.fractional_window + params.z_window - 1,
                       params.wave_window + 1, params.vol_window + 1))

    def indicators(self, data, params):
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if min(params.fractional_window, params.z_window,
               params.wave_window, params.vol_window) < 2:
            raise ValueError("indicator windows must be at least two")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five")
        if min(params.fd_threshold, params.wave_threshold,
               params.target_vol, params.profit_target) <= 0.0:
            raise ValueError("thresholds, target_vol and profit_target must be positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        # ponytail: fixed fractional memory; extend fractional_window for longer memory.
        weights = np.ones(params.fractional_window, dtype=float)
        for k in range(1, len(weights)):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        fractional = close.copy()
        for k in range(1, len(weights)):
            fractional = fractional + weights[k] * close.shift(k)
        out["fractional_close"] = fractional
        out["fd_mean"] = fractional.rolling(params.z_window).mean()
        out["fd_std"] = fractional.rolling(params.z_window).std(ddof=0)
        out["fd_z"] = (fractional - out["fd_mean"]) / out["fd_std"].replace(0.0, np.nan)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["range_overlap"] = (
            pd.concat([data["high"], out["previous_high"]], axis=1).min(axis=1, skipna=False)
            - pd.concat([data["low"], out["previous_low"]], axis=1).max(axis=1, skipna=False)
        ).clip(lower=0.0)
        out["range_union"] = (
            pd.concat([data["high"], out["previous_high"]], axis=1).max(axis=1, skipna=False)
            - pd.concat([data["low"], out["previous_low"]], axis=1).min(axis=1, skipna=False)
        ).replace(0.0, np.nan)
        out["range_midpoint"] = (data["high"] + data["low"]) / 2.0
        out["midpoint_step"] = out["range_midpoint"].diff()
        out["range_novelty"] = 1.0 - out["range_overlap"] / out["range_union"]
        out["signed_novelty"] = np.sign(out["midpoint_step"]) * out["range_novelty"]
        out["wave"] = out["signed_novelty"].rolling(params.wave_window).mean()
        out["log_return"] = np.log(close).diff()
        out["annual_vol"] = out["log_return"].rolling(params.vol_window).std(ddof=0) * np.sqrt(252.0)
        out["exposure"] = (
            params.target_vol / out["annual_vol"].replace(0.0, np.nan)
        ).clip(lower=0.05, upper=0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        n = len(data)
        candidate = np.select(
            [(indicators["fd_z"] > params.fd_threshold) & (indicators["wave"] > params.wave_threshold),
             (indicators["fd_z"] < -params.fd_threshold) & (indicators["wave"] < -params.wave_threshold)],
            [1, -1], default=0,
        ).astype(int)
        ready = np.isfinite(indicators[["fd_z", "wave", "annual_vol", "exposure"]]).all(axis=1).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        exposures = indicators["exposure"].fillna(0.05).clip(0.05, 0.95).to_numpy(dtype=float)
        signals = np.zeros(n, dtype=int)
        sizes = exposures.copy()
        position = 0
        entry_bar = -1
        entry_price = np.nan
        trade_size = 0.05
        for i in range(n):
            if position:
                # The previous close's decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry else np.nan
                exit_now = (not valid_entry or gain >= params.profit_target
                            or i - entry_bar + 1 >= params.hold_bars)
                sizes[i] = trade_size
                if exit_now:
                    position = 0
                    entry_price = np.nan
                else:
                    signals[i] = position
            elif (i >= self.warmup_bars(params) - 1 and i + 1 < n
                  and ready[i] and np.isfinite(closes[i]) and closes[i] > 0.0
                  and candidate[i] != 0):
                position = int(candidate[i])
                entry_bar = i + 1
                trade_size = float(exposures[i])
                signals[i] = position
                sizes[i] = trade_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = StrategyParams(fractional_window=2, z_window=2, wave_window=2, vol_window=2)
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"fd_z": 0.0, "wave": 0.0,
                               "annual_vol": 0.2, "exposure": 0.6}, index=index)
    for direction in (1, -1):
        test_indicators = indicators.copy()
        test_indicators.loc[index[2], ["fd_z", "wave"]] = [2.0 * direction, 0.5 * direction]
        result = strategy.generate_signals(data, test_indicators, None, params).data
        expected = np.zeros(12, dtype=int)
        expected[3:7] = direction
        assert np.array_equal(result["signal"].to_numpy(), expected)
        assert (result["size"] > 0.0).all()
        assert np.allclose(result["size"].iloc[3:7], 0.6)
        targeted = data.copy()
        targeted.loc[index[4], "close"] = 100.0 * (1.0 + direction * 0.02)
        result = strategy.generate_signals(targeted, test_indicators, None, params).data
        expected[5:7] = 0
        assert np.array_equal(result["signal"].to_numpy(), expected)
        test_indicators.loc[index[2], "wave"] *= -1.0
        assert not strategy.generate_signals(data, test_indicators, None, params).data["signal"].any()
    blank = indicators.copy()
    blank[["fd_z", "wave"]] = np.nan
    assert not strategy.generate_signals(data, blank, None, params).data["signal"].any()
    path_index = pd.date_range("2020-01-01", periods=100)
    path = 100.0 + np.arange(100) * 0.1 + np.sin(np.arange(100))
    history = pd.DataFrame({"open": path, "high": path + 1.0, "low": path - 1.0,
                            "close": path, "volume": 1000.0}, index=path_index)
    defaults = StrategyParams()
    full = strategy.indicators(history, defaults)
    assert full.iloc[:80].equals(strategy.indicators(history.iloc[:80], defaults))
    assert full.index.equals(history.index)
    assert np.isfinite(full[["fd_z", "wave", "annual_vol", "exposure"]].iloc[strategy.warmup_bars(defaults):]).all().all()


if __name__ == "__main__":
    _self_check()
