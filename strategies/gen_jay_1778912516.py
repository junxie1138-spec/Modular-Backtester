from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapFillRegimeParams:
    fill_window: int = 40
    fill_threshold: float = 0.55
    accum_window: int = 5
    accum_z_window: int = 60
    accum_z_thresh: float = 1.2
    atr_window: int = 14
    atr_stop_k: float = 2.5
    min_gap_pct: float = 0.001


class GeneratedStrategy(BaseStrategy[GapFillRegimeParams]):
    strategy_id = "gen_jay_1778912516"

    @classmethod
    def params_type(cls) -> type[GapFillRegimeParams]:
        return GapFillRegimeParams

    @staticmethod
    def warmup_bars(params: GapFillRegimeParams) -> int:
        return (
            max(
                params.fill_window,
                params.accum_z_window + params.accum_window,
                params.atr_window * 3,
            )
            + 5
        )

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapFillRegimeParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        prev_close = data["close"].shift(1)

        gap_raw = data["open"] - prev_close
        gap_pct = gap_raw / prev_close.replace(0.0, np.nan)

        gap_filled = (
            ((gap_raw > 0) & (data["close"] <= prev_close))
            | ((gap_raw < 0) & (data["close"] >= prev_close))
        )
        significant_gap = gap_pct.abs() > params.min_gap_pct

        sig_count = significant_gap.rolling(params.fill_window).sum().replace(0.0, np.nan)
        fill_count = (gap_filled & significant_gap).rolling(params.fill_window).sum()
        ind["fill_rate"] = fill_count / sig_count

        ind["trend_regime"] = (ind["fill_rate"] < params.fill_threshold).astype(float)

        ind["gap_accum"] = gap_pct.rolling(params.accum_window).sum()
        accum_mean = ind["gap_accum"].rolling(params.accum_z_window).mean()
        accum_std = ind["gap_accum"].rolling(params.accum_z_window).std()
        ind["gap_accum_z"] = (
            (ind["gap_accum"] - accum_mean) / accum_std.replace(0.0, np.nan)
        )

        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = true_range.ewm(span=params.atr_window, adjust=False).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapFillRegimeParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = 1.0

        gaz = indicators["gap_accum_z"].values
        fr = indicators["fill_rate"].values
        regime = indicators["trend_regime"].values
        atr = indicators["atr"].values
        close = data["close"].values
        n = len(df)

        valid = ~(np.isnan(gaz) | np.isnan(fr))
        trend_raw = np.where(
            gaz > params.accum_z_thresh, 1,
            np.where(gaz < -params.accum_z_thresh, -1, 0),
        )
        revert_raw = np.where(
            gaz > params.accum_z_thresh, -1,
            np.where(gaz < -params.accum_z_thresh, 1, 0),
        )
        raw = np.where(regime == 1, trend_raw, revert_raw)
        raw = np.where(valid, raw, 0).astype(int)

        raw_prev = np.empty_like(raw)
        raw_prev[0] = 0
        raw_prev[1:] = raw[:-1]
        confirmed = np.where((raw == raw_prev) & (raw != 0), raw, 0).astype(int)

        out = np.zeros(n, dtype=int)
        pos = 0
        entry_price = 0.0
        entry_atr_val = 0.0

        for i in range(n):
            if np.isnan(atr[i]):
                out[i] = 0
                continue

            if pos == 1:
                if close[i] <= entry_price - params.atr_stop_k * entry_atr_val:
                    pos = 0
                    out[i] = 0
                    continue
            elif pos == -1:
                if close[i] >= entry_price + params.atr_stop_k * entry_atr_val:
                    pos = 0
                    out[i] = 0
                    continue

            if confirmed[i] != 0 and confirmed[i] != pos:
                pos = confirmed[i]
                entry_price = close[i]
                entry_atr_val = atr[i]

            out[i] = pos

        df["signal"] = out
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
