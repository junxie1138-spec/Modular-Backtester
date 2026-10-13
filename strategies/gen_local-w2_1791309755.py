from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_window: int = 4
    regime_window: int = 20


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791309755"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return int(params.regime_window + params.path_window + 3)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        w = params.path_window
        r = params.regime_window
        if not isinstance(w, int) or not isinstance(r, int) or w < 3 or r < 3:
            raise ValueError("Both windows must be integers of at least three bars.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = np.log(close).diff()
        out["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["closing_pressure"] = ((2.0 * data["close"] - data["high"] - data["low"]) / out["range"]).clip(-1.0, 1.0)
        volume = data["volume"].where(data["volume"] >= 0)
        out["mean_volume"] = volume.rolling(r, min_periods=r).mean()
        out["relative_volume"] = volume / out["mean_volume"].where(out["mean_volume"] > 0)
        out["pressure"] = out["closing_pressure"] * out["relative_volume"]
        dx = out["return"]
        dy = out["pressure"]
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, w):
            cross = dx.shift(lag) * dy - dy.shift(lag) * dx
            area = area + cross.rolling(w - lag, min_periods=w - lag).sum()
        out["signature_area"] = 0.5 * area
        out["path_return"] = dx.rolling(w, min_periods=w).sum()
        out["path_pressure"] = dy.rolling(w, min_periods=w).sum()
        out["return_variation"] = dx.abs().rolling(w, min_periods=w).sum()
        out["pressure_variation"] = dy.abs().rolling(w, min_periods=w).sum()
        out["area_scale"] = out["return_variation"] * out["pressure_variation"]
        out["normalized_area"] = out["signature_area"] / out["area_scale"].where(out["area_scale"] > 0)
        out["area_score"] = out["normalized_area"].rolling(3, min_periods=3).mean()
        out["regime_return"] = dx.rolling(r, min_periods=r).sum()
        out["regime_variation"] = dx.abs().rolling(r, min_periods=r).sum()
        out["efficiency"] = out["regime_return"].abs() / out["regime_variation"].where(out["regime_variation"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        ready = indicators[["area_score", "path_return", "path_pressure", "efficiency", "regime_return"]].notna().all(axis=1)
        trending = indicators["efficiency"] >= 0.4
        continuation = trending & (indicators["regime_return"] > 0) & (indicators["path_return"] > 0)
        recovery = (~trending) & (indicators["path_return"] < 0)
        entry_condition = ready & (indicators["area_score"] < -0.08) & (indicators["path_pressure"] > 0) & (continuation | recovery)
        df = pd.DataFrame(index=data.index)
        # The desired position exits only when the active entry condition flips false.
        df["signal"] = entry_condition.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"area_score": [-0.2, -0.2, 0.1, -0.2, -0.2, -0.2, np.nan], "path_return": [0.01, 0.01, 0.01, -0.01, -0.01, -0.01, 0.01], "path_pressure": [1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0], "efficiency": [0.6, 0.6, 0.6, 0.2, 0.2, 0.6, 0.6], "regime_return": 0.02}, index=index)
    params = SignatureParams()
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 0, 0]
    assert (result["size"] > 0).all()
    flat = pd.concat([data] * 8, ignore_index=True)
    flat.index = pd.date_range("2020-01-01", periods=len(flat))
    derived = GeneratedStrategy.indicators(flat, params)
    assert not GeneratedStrategy.generate_signals(flat, derived, None, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
