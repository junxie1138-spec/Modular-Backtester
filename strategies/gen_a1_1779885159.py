from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapVolRegimeParams:
    vol_window: int = 20
    regime_ma: int = 200
    entry_z: float = 1.0
    news_threshold: float = 1.2
    noise_threshold: float = 0.8
    spike_z: float = 3.0
    refractory_bars: int = 5
    hold_bars: int = 18
    eps: float = 1e-9


class GeneratedStrategy(BaseStrategy[GapVolRegimeParams]):
    strategy_id = "gen_a1_1779885159"

    @classmethod
    def params_type(cls):
        return GapVolRegimeParams

    @classmethod
    def warmup_bars(cls, params: GapVolRegimeParams) -> int:
        return int(max(params.regime_ma, params.vol_window) + 2)

    def indicators(self, data: pd.DataFrame, params: GapVolRegimeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        prev_close = data["close"].shift(1)
        r_on = (data["open"] / prev_close) - 1.0
        r_id = (data["close"] / data["open"]) - 1.0
        gap_vol = r_on.rolling(params.vol_window, min_periods=params.vol_window).std()
        intra_vol = r_id.rolling(params.vol_window, min_periods=params.vol_window).std()
        gap_dominance = gap_vol / (intra_vol + params.eps)
        ma = data["close"].rolling(params.regime_ma, min_periods=params.regime_ma).mean()
        gap_z = r_on / (gap_vol + params.eps)
        out["r_on"] = r_on
        out["r_id"] = r_id
        out["gap_vol"] = gap_vol
        out["intra_vol"] = intra_vol
        out["gap_dominance"] = gap_dominance
        out["ma200"] = ma
        out["gap_z"] = gap_z
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapVolRegimeParams,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=np.int64)

        gap_z = indicators["gap_z"].to_numpy()
        gap_dom = indicators["gap_dominance"].to_numpy()
        ma = indicators["ma200"].to_numpy()
        close = data["close"].to_numpy()

        regime = np.zeros(n, dtype=np.int64)
        valid_ma = ~np.isnan(ma)
        regime[valid_ma & (close > ma)] = 1
        regime[valid_ma & (close < ma)] = -1

        in_pos = 0
        bars_in_pos = 0
        refractory_left = 0

        for i in range(n):
            if refractory_left > 0:
                refractory_left -= 1

            gz = gap_z[i]
            if not np.isnan(gz) and abs(gz) >= params.spike_z:
                refractory_left = params.refractory_bars

            if in_pos != 0:
                signal[i] = in_pos
                bars_in_pos += 1
                if bars_in_pos >= params.hold_bars:
                    in_pos = 0
                    bars_in_pos = 0
                continue

            if refractory_left > 0:
                continue

            gd = gap_dom[i]
            if np.isnan(gz) or np.isnan(gd) or regime[i] == 0:
                continue

            if abs(gz) < params.entry_z:
                continue

            gap_sign = 1 if gz > 0 else -1
            if gd > params.news_threshold:
                direction = gap_sign
            elif gd < params.noise_threshold:
                direction = -gap_sign
            else:
                continue

            if direction != regime[i]:
                continue

            in_pos = direction
            bars_in_pos = 1
            signal[i] = direction

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
