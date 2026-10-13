from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 6
    volatility_bars: int = 20
    center_bars: int = 10
    area_threshold: float = 0.10
    displacement_threshold: float = 0.75
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1790996635"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return int(max(params.path_bars + 1, params.volatility_bars + 1, params.center_bars))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        opening = data["open"].astype(float).where(data["open"] > 0)
        out["log_close"] = np.log(close)
        out["overnight"] = np.log(opening / close.shift(1))
        out["intraday"] = np.log(close / opening)
        out["return"] = out["log_close"].diff()
        out["volatility"] = out["return"].rolling(params.volatility_bars, min_periods=params.volatility_bars).std(ddof=0)
        out["center"] = out["log_close"].rolling(params.center_bars, min_periods=params.center_bars).mean()

        # Antisymmetric second-level signature: one term for every ordered pair.
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, params.path_bars):
            cross = out["overnight"].shift(lag) * out["intraday"] - out["intraday"].shift(lag) * out["overnight"]
            width = params.path_bars - lag
            area = area + cross.rolling(width, min_periods=width).sum()
        out["signature_area"] = 0.5 * area
        out["area_scale"] = (
            out["overnight"].abs().rolling(params.path_bars, min_periods=params.path_bars).sum()
            * out["intraday"].abs().rolling(params.path_bars, min_periods=params.path_bars).sum()
        )
        out["normalized_area"] = out["signature_area"] / out["area_scale"].where(out["area_scale"] > 0)
        out["displacement"] = (out["log_close"] - out["center"]) / out["volatility"].where(out["volatility"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        entry = (
            (indicators["normalized_area"] < -params.area_threshold)
            & (indicators["displacement"] < -params.displacement_threshold)
            & indicators[["normalized_area", "displacement"]].notna().all(axis=1)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        hold = max(1, int(params.hold_bars))
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    # Force a flat target at expiry; never renew on this bar.
                    exit_bar = -1
                continue
            if entry[i]:
                signal[i] = 1
                exit_bar = i + hold
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
