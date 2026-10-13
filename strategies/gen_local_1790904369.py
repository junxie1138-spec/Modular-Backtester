from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    ma_window: int = 20
    history_bars: int = 126
    shape_bars: int = 4
    bandwidth: float = 0.6
    min_analogs: int = 5
    entry_z: float = -0.75
    min_edge: float = 0.001
    pressure_window: int = 3
    min_pressure: float = 0.1
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790904369"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(
            params.ma_window + params.history_bars + params.shape_bars - 1,
            params.pressure_window,
        ))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        first_lag = params.shape_bars + params.hold_bars + 1
        if (
            params.ma_window < 2 or params.shape_bars < 2
            or params.hold_bars not in (1, 2)
            or params.history_bars < first_lag
            or not np.isfinite(params.bandwidth) or params.bandwidth <= 0
            or params.pressure_window < 1 or params.min_analogs < 1
            or params.min_analogs > params.history_bars - first_lag + 1
            or not np.isfinite(params.entry_z) or params.entry_z >= 0
            or not np.isfinite(params.min_edge) or params.min_edge < 0
            or not np.isfinite(params.min_pressure)
            or not 0 <= params.min_pressure < 1
        ):
            raise ValueError("Invalid analog strategy parameters")

        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["std"].replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        out["intraday_pressure"] = (
            (close - data["open"])
            / (data["high"] - data["low"]).replace(0, np.nan)
        ).clip(-1, 1)
        out["signed_volume"] = out["intraday_pressure"] * data["volume"]
        out["pressure_volume"] = data["volume"].rolling(params.pressure_window).sum()
        out["pressure"] = (
            out["signed_volume"].rolling(params.pressure_window).sum()
            / out["pressure_volume"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)

        weighted_return = pd.Series(0.0, index=data.index)
        weight_sum = pd.Series(0.0, index=data.index)
        analog_count = pd.Series(0, index=data.index, dtype=int)
        # ponytail: bounded lag scan; cache templates if histories become much longer.
        for lag in range(first_lag, params.history_bars + 1):
            distance = pd.Series(0.0, index=data.index)
            for offset in range(params.shape_bars):
                distance += (
                    out["z"].shift(offset) - out["z"].shift(lag + offset)
                ).pow(2) / params.shape_bars

            # Analog at t-lag: enter its next open, exit hold_bars opens later.
            # Both prices are in the past; no negative shifts are used.
            realized = (
                data["open"].shift(lag - params.hold_bars - 1)
                / data["open"].shift(lag - 1).replace(0, np.nan) - 1.0
            ).replace([np.inf, -np.inf], np.nan)
            match = distance.le(params.bandwidth ** 2) & realized.notna()
            weight = np.exp(-0.5 * distance / params.bandwidth ** 2).where(match, 0.0)
            weighted_return += weight * realized.fillna(0.0)
            weight_sum += weight
            analog_count += match.astype(int)

        out["analog_count"] = analog_count
        out["analog_weight"] = weight_sum
        out["analog_return_sum"] = weighted_return
        out["forecast"] = (
            weighted_return / weight_sum.replace(0, np.nan)
        ).where(analog_count >= params.min_analogs)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        # Primitive 1: a stretched z-score path with profitable matured analogs.
        analog_agrees = (
            indicators["z"].le(params.entry_z)
            & indicators["forecast"].gt(params.min_edge)
            & indicators["analog_count"].ge(params.min_analogs)
        )
        # Primitive 2: independent volume-weighted intraday buying pressure.
        pressure_agrees = indicators["pressure"].gt(params.min_pressure)
        eligible = (analog_agrees & pressure_agrees).fillna(False).to_numpy(dtype=bool)
        eligible[:self.warmup_bars(params)] = False
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    # Force one flat target; entry conditions cannot postpone expiry.
                    exit_bar = -1
            elif eligible[i]:
                signal[i] = 1
                exit_bar = i + params.hold_bars

        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = AnalogParams()
    warmup = strategy.warmup_bars(params)
    t = np.arange(warmup + 40, dtype=float)
    close = 100.0 + 0.03 * t + 2.0 * np.sin(t / 3.0)
    opening = close + 0.2 * np.cos(t)
    data = pd.DataFrame({
        "open": opening,
        "high": np.maximum(opening, close) + 0.5,
        "low": np.minimum(opening, close) - 0.5,
        "close": close,
        "volume": 1000000.0 + 10000.0 * np.sin(t),
    }, index=pd.date_range("2020-01-01", periods=len(t), freq="B"))
    derived = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(
        derived.iloc[:-7], strategy.indicators(data.iloc[:-7], params)
    )
    forced = derived.copy()
    forced["z"] = -2.0
    forced["forecast"] = 0.01
    forced["analog_count"] = 10
    forced["pressure"] = 0.5
    ctx = StrategyContext("SPY", "1d", warmup)
    for holding in (1, 2):
        p = AnalogParams(hold_bars=holding)
        result = strategy.generate_signals(data, forced, ctx, p).data
        expected = [0] + [1] * holding + [0] + [1] * holding + [0]
        assert result["signal"].iloc[warmup:warmup + len(expected)].tolist() == expected
        assert result["signal"].isin([0, 1]).all()
        assert result["size"].gt(0).all()
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    flat["volume"] = 0.0
    result = strategy.generate_signals(
        flat, strategy.indicators(flat, params), ctx, params
    ).data
    assert result["signal"].eq(0).all()
