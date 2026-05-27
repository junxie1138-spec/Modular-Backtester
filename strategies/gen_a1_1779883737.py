from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SpringTensionParams:
    energy_window: int = 10
    energy_percentile_window: int = 60
    energy_threshold: float = 0.85
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    ma_period: int = 200
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[SpringTensionParams]):
    strategy_id = "gen_a1_1779883737"

    @classmethod
    def params_type(cls) -> type[SpringTensionParams]:
        return SpringTensionParams

    def warmup_bars(self, params: SpringTensionParams) -> int:
        return max(params.ma_period, params.energy_percentile_window + params.energy_window, params.atr_period) + 2

    def indicators(self, data: pd.DataFrame, params: SpringTensionParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        volume = data["volume"].astype(float)

        ret = close.pct_change()

        vol_mean = volume.rolling(params.energy_window, min_periods=params.energy_window).mean()
        vol_mean_safe = vol_mean.where(vol_mean > 0, np.nan)
        vol_norm = volume / vol_mean_safe

        bar_signed = ret * vol_norm
        bar_energy = (ret * ret) * vol_norm

        energy = bar_energy.rolling(params.energy_window, min_periods=params.energy_window).sum()
        signed_sum = bar_signed.rolling(params.energy_window, min_periods=params.energy_window).sum()
        direction = np.sign(signed_sum)

        energy_rank = energy.rolling(
            params.energy_percentile_window,
            min_periods=params.energy_percentile_window,
        ).rank(pct=True)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        ma = close.rolling(params.ma_period, min_periods=params.ma_period).mean()

        latest_sign = np.sign(bar_signed)

        out = pd.DataFrame(index=data.index)
        out["energy"] = energy
        out["energy_rank"] = energy_rank
        out["direction"] = direction
        out["latest_sign"] = latest_sign
        out["atr"] = atr
        out["ma"] = ma
        out["bar_signed"] = bar_signed
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SpringTensionParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        n = len(close)

        energy_rank = indicators["energy_rank"].to_numpy(dtype=float)
        direction = indicators["direction"].to_numpy(dtype=float)
        latest_sign = indicators["latest_sign"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)

        raw = np.zeros(n, dtype=np.int64)
        for i in range(n):
            er = energy_rank[i]
            d = direction[i]
            ls = latest_sign[i]
            m = ma[i]
            a = atr[i]
            c = close[i]
            if np.isnan(er) or np.isnan(d) or np.isnan(ls) or np.isnan(m) or np.isnan(a):
                continue
            if er <= params.energy_threshold:
                continue
            if d == 0.0 or ls == 0.0 or d != ls:
                continue
            if d > 0.0 and c > m:
                raw[i] = 1
            elif d < 0.0 and c < m:
                raw[i] = -1

        signal = np.zeros(n, dtype=np.int64)
        position = 0
        entry_price = np.nan
        entry_atr = np.nan
        bars_held = 0

        for i in range(n):
            if position == 0:
                if raw[i] != 0 and not np.isnan(atr[i]):
                    position = int(raw[i])
                    entry_price = close[i]
                    entry_atr = atr[i]
                    bars_held = 0
                    signal[i] = position
                else:
                    signal[i] = 0
            else:
                bars_held += 1
                stop_hit = False
                if position == 1:
                    if close[i] < entry_price - params.atr_stop_mult * entry_atr:
                        stop_hit = True
                else:
                    if close[i] > entry_price + params.atr_stop_mult * entry_atr:
                        stop_hit = True
                if stop_hit or bars_held >= params.max_hold:
                    position = 0
                    entry_price = np.nan
                    entry_atr = np.nan
                    bars_held = 0
                    signal[i] = 0
                else:
                    signal[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
