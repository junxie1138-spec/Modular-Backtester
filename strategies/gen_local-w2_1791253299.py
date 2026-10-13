from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    baseline_window: int = 20
    scale_window: int = 10
    compression_ratio: float = 0.8
    area_threshold: float = 0.15
    pressure_threshold: float = 0.3


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791253299"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(params.baseline_window + 1, params.scale_window + 1, 3))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if params.baseline_window < 2 or params.scale_window < 2:
            raise ValueError("Windows must be at least two bars.")
        if not 0.0 < params.compression_ratio < 1.0:
            raise ValueError("Compression ratio must be between zero and one.")
        if not np.isfinite(params.area_threshold) or params.area_threshold <= 0.0:
            raise ValueError("Area threshold must be positive and finite.")
        if not 0.0 < params.pressure_threshold < 1.0:
            raise ValueError("Pressure threshold must be between zero and one.")

        out = pd.DataFrame(index=data.index)
        out["close_positive"] = data["close"].where(data["close"] > 0.0)
        out["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["relative_range"] = out["range"] / out["close_positive"]
        out["log_price"] = np.log(out["close_positive"])
        out["log_range"] = np.log(out["relative_range"])
        out["price_step"] = out["log_price"].diff()
        out["range_step"] = out["log_range"].diff()
        out["price_scale"] = out["price_step"].abs().rolling(params.scale_window, min_periods=params.scale_window).mean()
        out["range_scale"] = out["range_step"].abs().rolling(params.scale_window, min_periods=params.scale_window).mean()
        out["area_scale"] = (out["price_scale"] * out["range_scale"]).replace(0.0, np.nan)
        # Antisymmetric second-order signature for two consecutive path increments.
        out["signature_area"] = 0.5 * (
            out["price_step"].shift(1) * out["range_step"]
            - out["range_step"].shift(1) * out["price_step"]
        ) / out["area_scale"]
        out["range_baseline"] = out["relative_range"].shift(1).rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        out["compression"] = out["relative_range"] / out["range_baseline"].replace(0.0, np.nan)
        out["closing_pressure"] = ((2.0 * data["close"] - data["high"] - data["low"]) / out["range"]).clip(-1.0, 1.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        valid = indicators[["signature_area", "compression", "closing_pressure"]].notna().all(axis=1)
        compressed = valid & (indicators["compression"] < params.compression_ratio)
        bullish = compressed & (indicators["signature_area"] < -params.area_threshold) & (indicators["closing_pressure"] > params.pressure_threshold)
        bearish = compressed & (indicators["signature_area"] > params.area_threshold) & (indicators["closing_pressure"] < -params.pressure_threshold)
        entry = np.select([bullish, bearish], [1, -1], default=0).astype(int)

        position = 0
        states = np.zeros(len(data), dtype=int)
        for i, direction in enumerate(entry):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                # Opposite agreement exits; a persistent opposite setup can enter next bar.
                position = 0
            states[i] = position

        df = pd.DataFrame({"signal": states, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = SignatureParams()
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({
        "signature_area": [-1.0, np.nan, 1.0, 1.0, np.nan, -1.0],
        "compression": [0.5] * 6,
        "closing_pressure": [0.8, 0.0, -0.8, -0.8, 0.0, 0.8],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, derived, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    assert result.data["size"].gt(0.0).all()
    assert strategy.warmup_bars(params) >= params.baseline_window + 1
    assert strategy.indicators(data, params)["signature_area"].isna().all()


if __name__ == "__main__":
    _self_check()
