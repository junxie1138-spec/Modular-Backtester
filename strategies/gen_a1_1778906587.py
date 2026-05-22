from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StandingWaveParams:
    er_window: int = 10
    pct_window: int = 100
    atr_window: int = 14
    vol_window: int = 20
    sma_window: int = 100
    entry_pct: float = 0.75
    k_atr: float = 2.5
    max_hold: int = 5
    target_vol: float = 0.012
    min_size: float = 0.3
    size_cap: float = 1.5


class GeneratedStrategy(BaseStrategy[StandingWaveParams]):
    strategy_id = "gen_a1_1778906587"

    @classmethod
    def params_type(cls) -> type[StandingWaveParams]:
        return StandingWaveParams

    @staticmethod
    def warmup_bars(params: StandingWaveParams) -> int:
        return int(
            max(
                params.er_window + params.pct_window,
                params.atr_window,
                params.vol_window,
                params.sma_window,
            )
            + 5
        )

    def indicators(self, data: pd.DataFrame, params: StandingWaveParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        out = pd.DataFrame(index=data.index)

        # True range / ATR -- used for the fixed volatility stop.
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr"] = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        # Path-efficiency ratio: net displacement / total absolute path travelled.
        # High ratio == traveling wave; low ratio == standing (oscillating) wave.
        disp = (close - close.shift(params.er_window)).abs()
        path = close.diff().abs().rolling(
            params.er_window, min_periods=params.er_window
        ).sum()
        eff = disp / path.replace(0.0, np.nan)

        # Twist: percentile threshold -- rank the efficiency ratio within its own
        # rolling history rather than testing a fixed efficiency level.
        out["eff_pct"] = eff.rolling(
            params.pct_window, min_periods=params.pct_window
        ).rank(pct=True)

        # Directional confirmation: net move over the efficiency window is up.
        out["up_dir"] = (close > close.shift(params.er_window)).astype(float)

        # Momentum-family trend gate.
        sma = close.rolling(params.sma_window, min_periods=params.sma_window).mean()
        out["trend_ok"] = (close > sma).astype(float)

        # Volatility-targeted position size from realized return volatility.
        ret = close.pct_change()
        rv = ret.rolling(params.vol_window, min_periods=params.vol_window).std()
        vol_size = (params.target_vol / rv.replace(0.0, np.nan)).clip(
            lower=params.min_size, upper=params.size_cap
        )
        out["vol_size"] = vol_size.fillna(1.0)

        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StandingWaveParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        eff_pct = indicators["eff_pct"].to_numpy(dtype=float)
        up_dir = indicators["up_dir"].to_numpy(dtype=float)
        trend_ok = indicators["trend_ok"].to_numpy(dtype=float)
        vol_size = indicators["vol_size"].to_numpy(dtype=float)

        raw_signal = np.zeros(n, dtype=int)
        size_raw = np.ones(n, dtype=float)

        in_pos = False
        entry_price = 0.0
        entry_atr = 0.0
        entry_size = 1.0
        bars_held = 0

        for i in range(n):
            if np.isnan(atr[i]) or np.isnan(eff_pct[i]):
                continue

            if in_pos:
                bars_held += 1
                stop_level = entry_price - params.k_atr * entry_atr
                hit_stop = close[i] < stop_level
                time_out = bars_held >= params.max_hold
                if hit_stop or time_out:
                    in_pos = False
                    bars_held = 0
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = 1
                    size_raw[i] = entry_size
            else:
                entry_cond = (
                    eff_pct[i] >= params.entry_pct
                    and up_dir[i] >= 0.5
                    and trend_ok[i] >= 0.5
                )
                if entry_cond:
                    in_pos = True
                    bars_held = 0
                    entry_price = close[i]
                    entry_atr = atr[i]
                    s = vol_size[i]
                    if not np.isfinite(s) or s <= 0.0:
                        s = 1.0
                    entry_size = float(s)
                    raw_signal[i] = 1
                    size_raw[i] = entry_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = (
            pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        )
        size_series = pd.Series(size_raw, index=data.index).shift(1).fillna(1.0)
        size_series = size_series.clip(lower=params.min_size, upper=params.size_cap)
        df["size"] = size_series.astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
