from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureRiskParams:
    area_threshold: float = 0.15
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[SignatureRiskParams]):
    strategy_id = "gen_local_1790928133"

    @classmethod
    def params_type(cls) -> type[SignatureRiskParams]:
        return SignatureRiskParams

    def warmup_bars(self, params: SignatureRiskParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: SignatureRiskParams) -> pd.DataFrame:
        if not 0.0 < params.area_threshold < 1.0 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("Require 0 < area_threshold < 1 and finite stop_atr > 0.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["variance_increment"] = out["return"].pow(2)
        out["daily_vol"] = out["return"].rolling(20, min_periods=20).std(ddof=0)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["net_return"] = out["return"].rolling(5, min_periods=5).sum()
        out["absolute_return_sum"] = out["return"].abs().rolling(5, min_periods=5).sum()
        out["variance_sum"] = out["variance_increment"].rolling(5, min_periods=5).sum()
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, 5):
            cross = (
                out["return"].shift(lag) * out["variance_increment"]
                - out["variance_increment"].shift(lag) * out["return"]
            )
            area = area + cross.rolling(5 - lag, min_periods=5 - lag).sum()
        out["signature_area"] = 0.5 * area
        denominator = out["absolute_return_sum"] * out["variance_sum"]
        out["normalized_area"] = (2.0 * out["signature_area"] / denominator.where(denominator > 0.0)).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureRiskParams,
    ) -> SignalFrame:
        if not 0.0 < params.area_threshold < 1.0 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("Require 0 < area_threshold < 1 and finite stop_atr > 0.")
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.05, dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        area = indicators["normalized_area"].to_numpy(dtype=float)
        net = indicators["net_return"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        vol = indicators["daily_vol"].to_numpy(dtype=float)
        side, age = 0, 0
        entry, distance, exposure = 0.0, 0.0, 0.05
        for i in range(n):
            if side:
                # The previous close's decision fills at this session's open.
                if age == 0:
                    if np.isfinite(opening[i]) and opening[i] > 0.0:
                        entry = opening[i]
                    else:
                        side = 0
                if side:
                    age += 1
                    stopped = np.isfinite(closing[i]) and side * (closing[i] - entry) < -distance
                    if stopped or age >= 4:
                        side, age = 0, 0
            elif (
                np.isfinite(area[i]) and np.isfinite(net[i])
                and np.isfinite(atr[i]) and atr[i] > 0.0
                and np.isfinite(vol[i]) and vol[i] > 0.0
                and np.isfinite(closing[i]) and closing[i] > 0.0
            ):
                if area[i] < -params.area_threshold and net[i] > 0.0:
                    side = 1
                elif area[i] > params.area_threshold and net[i] < 0.0:
                    side = -1
                if side:
                    age = 0
                    distance = params.stop_atr * atr[i]
                    exposure = float(np.clip(0.10 / (vol[i] * np.sqrt(252.0)), 0.05, 0.95))
            signals[i] = side
            sizes[i] = exposure
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=14, freq="B")
    data = pd.DataFrame({
        "open": 100.0, "high": 104.0, "low": 96.0,
        "close": 100.0, "volume": 1000000.0,
    }, index=index)
    data.loc[index[[2, 3, 6, 7]], "close"] = [99.0, 97.0, 101.0, 103.0]
    indicators = pd.DataFrame({
        "normalized_area": 0.0, "net_return": 0.0,
        "atr": 0.1, "daily_vol": 0.02,
    }, index=index)
    indicators.loc[index[[0, 4, 8]], "normalized_area"] = [-0.4, 0.4, -0.4]
    indicators.loc[index[[0, 4, 8]], "net_return"] = [0.01, -0.01, 0.01]
    indicators.loc[index[[0, 4, 8]], "atr"] = 1.0
    strategy = GeneratedStrategy()
    params = SignatureRiskParams()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0, 1, 1, 1, 1, 0]
    assert np.isfinite(result["size"]).all() and (result["size"] > 0.0).all()
    prefix = strategy.generate_signals(data.iloc[:11], indicators.iloc[:11], None, params).data
    pd.testing.assert_frame_equal(prefix, result.iloc[:11])
    flat = strategy.indicators(data.assign(close=100.0), params)
    assert strategy.generate_signals(data, flat, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
