from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class YieldStressParams:
    ma_window: int = 20
    rank_window: int = 252
    yield_percentile: float = 0.85
    atr_window: int = 14
    atr_stop_mult: float = 2.5
    max_hold_bars: int = 20
    regime_ma: int = 200
    use_regime_filter: bool = True


class GeneratedStrategy(BaseStrategy[YieldStressParams]):
    strategy_id = "gen_a1_1779884541"

    @classmethod
    def params_type(cls) -> type[YieldStressParams]:
        return YieldStressParams

    def warmup_bars(self, params: YieldStressParams) -> int:
        return int(max(
            params.ma_window,
            params.rank_window,
            params.atr_window + 1,
            params.regime_ma,
        )) + 2

    def indicators(self, data: pd.DataFrame, params: YieldStressParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        sma = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        std = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        std_safe = std.where(std > 0.0)
        zscore = (close - sma) / std_safe

        z_rank = zscore.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        regime_sma = close.rolling(params.regime_ma, min_periods=params.regime_ma).mean()
        regime_ok = (close > regime_sma).astype(float)
        regime_ok = regime_ok.where(regime_sma.notna(), other=np.nan)

        out = pd.DataFrame({
            "zscore": zscore,
            "z_rank": z_rank,
            "atr": atr,
            "regime_ok": regime_ok,
        }, index=data.index)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: YieldStressParams,
    ) -> SignalFrame:
        close_arr = data["close"].to_numpy(dtype=np.float64)
        z_rank_arr = indicators["z_rank"].to_numpy(dtype=np.float64)
        atr_arr = indicators["atr"].to_numpy(dtype=np.float64)
        regime_arr = indicators["regime_ok"].to_numpy(dtype=np.float64)

        n = len(close_arr)
        raw_signal = np.zeros(n, dtype=np.int64)
        size_arr = np.ones(n, dtype=np.float64)

        threshold = float(params.yield_percentile)
        stop_mult = float(params.atr_stop_mult)
        max_hold = int(params.max_hold_bars)
        use_regime = bool(params.use_regime_filter)

        in_pos = False
        entry_idx = -1
        hwm = -np.inf
        prev_rank = np.nan

        for i in range(n):
            zr = z_rank_arr[i]
            a = atr_arr[i]
            c = close_arr[i]
            r_ok = regime_arr[i]

            if in_pos:
                if c > hwm:
                    hwm = c

                exit_now = False
                if (not np.isnan(a)) and a > 0.0 and c <= hwm - stop_mult * a:
                    exit_now = True
                if (i - entry_idx) >= max_hold:
                    exit_now = True

                if exit_now:
                    raw_signal[i] = 0
                    in_pos = False
                    entry_idx = -1
                    hwm = -np.inf
                else:
                    raw_signal[i] = 1
            else:
                crossed = (
                    (not np.isnan(zr))
                    and (not np.isnan(prev_rank))
                    and prev_rank <= threshold
                    and zr > threshold
                )
                regime_pass = (not use_regime) or (
                    (not np.isnan(r_ok)) and r_ok > 0.5
                )
                atr_ok = (not np.isnan(a)) and a > 0.0

                if crossed and regime_pass and atr_ok:
                    raw_signal[i] = 1
                    in_pos = True
                    entry_idx = i
                    hwm = c
                else:
                    raw_signal[i] = 0

            prev_rank = zr

        signal_series = pd.Series(raw_signal, index=data.index)
        signal_series = signal_series.shift(1).fillna(0).astype(int)
        size_series = pd.Series(size_arr, index=data.index)

        df = pd.DataFrame({
            "signal": signal_series.values,
            "size": size_series.values,
        }, index=data.index)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
