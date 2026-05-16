from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    gap_window: int = 15
    atr_short: int = 7
    atr_long: int = 63
    entropy_entry: float = 0.85
    entropy_exit: float = 0.92
    compression_thresh: float = 0.85
    min_pos_frac: float = 0.60
    base_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778914356"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.atr_long + params.gap_window + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        prev_close = data["close"].shift(1)
        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr_s = tr.rolling(params.atr_short).mean()
        atr_l = tr.rolling(params.atr_long).mean()
        ind["range_compression"] = atr_s / atr_l.replace(0, np.nan)

        gap_sign = np.sign(data["open"] - prev_close)
        raw_p = (gap_sign > 0).rolling(params.gap_window).mean()
        ind["pos_frac"] = raw_p
        p_safe = raw_p.clip(1e-9, 1 - 1e-9)
        ind["gap_entropy"] = -(p_safe * np.log2(p_safe) + (1 - p_safe) * np.log2(1 - p_safe))

        entropy_strength = (
            (params.entropy_entry - ind["gap_entropy"]).clip(lower=0) / params.entropy_entry
        )
        compression_strength = (
            (params.compression_thresh - ind["range_compression"]).clip(lower=0)
            / params.compression_thresh
        )
        combined = (entropy_strength + compression_strength) / 2.0
        ind["size_signal"] = (params.base_size * (0.6 + 0.4 * combined)).clip(0.3, 1.0)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        sig = np.zeros(n, dtype=int)
        sz = np.full(n, params.base_size)

        rc = indicators["range_compression"].to_numpy()
        pf = indicators["pos_frac"].to_numpy()
        ge = indicators["gap_entropy"].to_numpy()
        ss = indicators["size_signal"].to_numpy()

        in_trade = False

        for i in range(n):
            if np.isnan(rc[i]) or np.isnan(pf[i]) or np.isnan(ge[i]):
                continue

            entry_cond = (
                rc[i] < params.compression_thresh
                and ge[i] < params.entropy_entry
                and pf[i] >= params.min_pos_frac
            )
            exit_cond = ge[i] > params.entropy_exit or pf[i] < 0.5

            if not in_trade and entry_cond:
                in_trade = True
            elif in_trade and exit_cond:
                in_trade = False

            if in_trade:
                sig[i] = 1
                sz[i] = ss[i] if not np.isnan(ss[i]) else params.base_size

        df = pd.DataFrame({"signal": sig, "size": sz}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.base_size)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
