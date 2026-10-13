from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReleaseParams:
    shock_ratio: float = 1.35
    calm_ratio: float = 0.8
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[ReleaseParams]):
    strategy_id = "gen_local-w2_1791296531"

    @classmethod
    def params_type(cls) -> type[ReleaseParams]:
        return ReleaseParams

    @classmethod
    def warmup_bars(cls, params: ReleaseParams) -> int:
        return 200

    def indicators(self, data: pd.DataFrame, params: ReleaseParams) -> pd.DataFrame:
        close = data["close"]
        returns = np.log(close.clip(lower=1e-12)).diff()
        fast_vol = returns.rolling(4, min_periods=4).std()
        slow_vol = returns.rolling(30, min_periods=30).std()
        ratio = fast_vol.div(slow_vol.replace(0, np.nan))
        peak = ratio.shift(1).rolling(8, min_periods=8).max()
        ma200 = close.rolling(200, min_periods=200).mean()
        previous_close = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean()
        ready = ma200.notna() & atr.notna() & ratio.notna() & peak.notna()
        return pd.DataFrame({
            "ma200": ma200.fillna(0.0),
            "atr": atr.fillna(0.0),
            "vol_ratio": ratio.fillna(0.0),
            "recent_peak": peak.fillna(0.0),
            "ready": ready,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ReleaseParams,
    ) -> SignalFrame:
        ratio = indicators["vol_ratio"]
        entries = (
            indicators["ready"]
            & (data["close"] > indicators["ma200"])
            & (ratio < params.calm_ratio)
            & (ratio.shift(1).fillna(0.0) >= params.calm_ratio)
            & (indicators["recent_peak"] >= params.shock_ratio)
        ).to_numpy()
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        ma200 = indicators["ma200"].to_numpy()
        atr = indicators["atr"].to_numpy()
        signal = np.zeros(len(data), dtype=int)
        held = False
        pending_entry = False
        stop = 0.0
        entry_bar = 0

        for i in range(len(data)):
            if pending_entry:
                held = True
                stop = float(opens[i]) - params.stop_atr * float(atr[i - 1])
                entry_bar = i
                pending_entry = False
            if held:
                if closes[i] < stop or i - entry_bar >= params.hold_bars or closes[i] <= ma200[i]:
                    held = False
                else:
                    signal[i] = 1
            elif entries[i]:
                signal[i] = 1
                pending_entry = True

        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
