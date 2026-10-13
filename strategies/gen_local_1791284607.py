from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EquilibriumParams:
    channel_window: int = 20
    energy_window: int = 5
    baseline_window: int = 20
    energy_ratio: float = 0.8
    entropy_floor: float = 0.8
    atr_window: int = 14
    trailing_atr: float = 1.0
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[EquilibriumParams]):
    strategy_id = "gen_local_1791284607"

    @classmethod
    def params_type(cls) -> type[EquilibriumParams]:
        return EquilibriumParams

    def warmup_bars(self, params: EquilibriumParams) -> int:
        return int(max(
            8 + params.energy_window - 1 + params.baseline_window - 1,
            params.channel_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: EquilibriumParams) -> pd.DataFrame:
        if min(params.channel_window, params.energy_window, params.baseline_window, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two bars")
        if not (0.0 < params.energy_ratio and 0.0 <= params.entropy_floor <= 1.0):
            raise ValueError("Invalid energy thresholds")
        if not np.isfinite(params.energy_ratio) or not np.isfinite(params.trailing_atr) or params.trailing_atr <= 0.0:
            raise ValueError("Energy ratio and ATR multiplier must be finite and positive")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("Holding horizon must be one or two sessions")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0.0)).replace([np.inf, -np.inf], np.nan)
        energy_columns = []
        for scale in (2, 4, 8):
            half = scale // 2
            block_mean = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"haar_{scale}"] = (block_mean - block_mean.shift(half)) * np.sqrt(half / 2.0)
            energy_name = f"energy_{scale}"
            out[energy_name] = out[f"haar_{scale}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
            energy_columns.append(energy_name)

        energies = out[energy_columns]
        out["total_energy"] = energies.sum(axis=1, min_count=3)
        weights = energies.div(out["total_energy"].replace(0.0, np.nan), axis=0)
        out["energy_entropy"] = -(weights * np.log(weights.clip(lower=1e-12))).sum(
            axis=1, min_count=3
        ) / np.log(3.0)
        out["energy_baseline"] = out["total_energy"].rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).median()
        out["wavelet_agrees"] = (
            (out["total_energy"] > 0.0)
            & (out["total_energy"] <= params.energy_ratio * out["energy_baseline"])
            & (out["energy_entropy"] >= params.entropy_floor)
        ).fillna(False)

        out["channel_high"] = data["high"].rolling(
            params.channel_window, min_periods=params.channel_window
        ).max()
        out["channel_low"] = data["low"].rolling(
            params.channel_window, min_periods=params.channel_window
        ).min()
        width = (out["channel_high"] - out["channel_low"]).where(
            out["channel_high"] > out["channel_low"]
        )
        out["relative_position"] = (close - out["channel_low"]) / width
        out["position_agrees"] = (
            (out["relative_position"].shift(1) <= 0.5)
            & (out["relative_position"] > 0.5)
            & (out["relative_position"] <= 0.75)
        ).fillna(False)

        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EquilibriumParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = (
            indicators["wavelet_agrees"].fillna(False).astype(bool)
            & indicators["position_agrees"].fillna(False).astype(bool)
        ).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = False
        held_bars = 0
        high_water = -np.inf
        stop = -np.inf

        for i in range(len(data)):
            price = close[i]
            volatility = atr[i]
            valid = np.isfinite(price) and price > 0.0 and np.isfinite(volatility) and volatility > 0.0
            if active:
                if not valid:
                    active = False
                    continue
                # The previous decision fills on this bar: only now track in-trade closes.
                held_bars += 1
                high_water = max(high_water, price)
                stop = max(stop, high_water - params.trailing_atr * volatility)
                if price <= stop or held_bars >= params.max_hold_bars:
                    active = False
                else:
                    raw[i] = 1
            elif valid and entries[i]:
                active = True
                held_bars = 0
                high_water = -np.inf
                stop = -np.inf
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    data = pd.DataFrame({
        "open": [100.0, 101.0, 104.0, 104.0],
        "high": [101.0, 106.0, 107.0, 108.0],
        "low": [99.0, 100.0, 102.0, 103.0],
        "close": [100.0, 105.0, 106.0, 107.0],
        "volume": [1000.0] * 4,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0] * 4,
        "wavelet_agrees": [True, False, False, False],
        "position_agrees": [True, False, False, False],
    }, index=index)
    params = EquilibriumParams()
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0]
    # Isolate the ATR exit from expiry: ATR expansion must never loosen the stop.
    params.max_hold_bars = 10
    data["close"] = [100.0, 105.0, 103.0, 104.0]
    indicators["atr"] = [1.0, 1.0, 10.0, 1.0]
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0]
    indicators["position_agrees"] = False
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].eq(0).all()
