from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    lookback: int = 20
    smoothing: int = 5
    area_floor: float = 0.002
    position_floor: float = 0.55


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791262583"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(params.lookback + params.smoothing)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        w, s = params.lookback, params.smoothing
        if w < 2 or s < 1:
            raise ValueError("lookback must be >= 2 and smoothing >= 1")
        out = pd.DataFrame(index=data.index)
        midpoint = (data["high"] + data["low"]) / 2.0
        out["log_midpoint"] = np.log(midpoint.where(midpoint > 0))
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        x, y = out["log_midpoint"], out["log_close"]
        out["dx"], out["dy"] = x.diff(), y.diff()
        out["cross_increment"] = x.shift(1) * out["dy"] - y.shift(1) * out["dx"]
        out["x_start"], out["y_start"] = x.shift(w), y.shift(w)
        out["x_variation"] = out["dx"].abs().rolling(w, min_periods=w).sum()
        out["y_variation"] = out["dy"].abs().rolling(w, min_periods=w).sum()
        out["cross_sum"] = out["cross_increment"].rolling(w, min_periods=w).sum()
        # Antisymmetric level-two signature of the translated midpoint-close path.
        out["signed_area"] = 0.5 * (
            out["cross_sum"] - out["x_start"] * (y - out["y_start"])
            + out["y_start"] * (x - out["x_start"])
        )
        out["area_scale"] = out["x_variation"] * out["y_variation"]
        out["normalized_area"] = out["signed_area"] / out["area_scale"].where(out["area_scale"] > 0)
        out["signature_area"] = out["normalized_area"].rolling(s, min_periods=s).mean()
        out["range_low"] = data["low"].rolling(w, min_periods=w).min()
        out["range_high"] = data["high"].rolling(w, min_periods=w).max()
        out["range_width"] = out["range_high"] - out["range_low"]
        out["relative_position"] = (
            (data["close"] - out["range_low"]) / out["range_width"].where(out["range_width"] > 0)
        )
        out["position_mean"] = out["relative_position"].rolling(s, min_periods=s).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SignatureParams,
    ) -> SignalFrame:
        valid = np.isfinite(indicators["signature_area"]) & np.isfinite(indicators["position_mean"])
        # Hold only while BOTH primitives agree; their conjunction reversing is the only exit.
        entry = (
            valid
            & (indicators["signature_area"] > params.area_floor)
            & (indicators["position_mean"] > params.position_floor)
        )
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy, params = GeneratedStrategy(), SignatureParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    indicators = pd.DataFrame({
        "signature_area": [np.nan, 0.01, 0.01, -0.01, 0.01, 0.01],
        "position_mean": [0.8, 0.8, 0.3, 0.8, 0.8, 0.8],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, 1]
    assert (result["size"] > 0).all()
    flat = pd.DataFrame(100.0, index=pd.date_range("2020-01-01", periods=40),
                        columns=data.columns)
    result = strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params).data
    assert result["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
