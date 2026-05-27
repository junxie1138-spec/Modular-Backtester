from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    atr_period: int = 14
    strain_ma_period: int = 20
    yield_window: int = 252
    yield_percentile: float = 0.80
    elastic_threshold: float = 1.5
    plastic_threshold: float = 1.0
    regime_ma_period: int = 200
    profit_target_pct: float = 0.02
    time_stop_bars: int = 2
    size_pct: float = 1.0


class GeneratedStrategy(BaseStrategy):
    strategy_id = "gen_a1_1779878574"

    @classmethod
    def params_type(cls):
        return StrategyParams

    @classmethod
    def warmup_bars(cls, params):
        return int(
            max(params.yield_window, params.regime_ma_period)
            + max(params.atr_period, params.strain_ma_period)
            + 5
        )

    def indicators(self, data, params):
        close = data["close"]
        high = data["high"]
        low = data["low"]
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

        sma_strain = close.rolling(
            params.strain_ma_period, min_periods=params.strain_ma_period
        ).mean()
        atr_safe = atr.replace(0.0, np.nan)
        strain = (close - sma_strain) / atr_safe

        yield_strength = atr.rolling(
            params.yield_window, min_periods=params.yield_window
        ).quantile(params.yield_percentile)

        ma200 = close.rolling(
            params.regime_ma_period, min_periods=params.regime_ma_period
        ).mean()

        ind = pd.DataFrame(
            {
                "atr": atr,
                "strain": strain,
                "yield_strength": yield_strength,
                "ma200": ma200,
                "ref_close": close,
            },
            index=data.index,
        )
        return ind

    def generate_signals(self, data, indicators, ctx, params):
        close = indicators["ref_close"].to_numpy(dtype=float)
        strain = indicators["strain"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        yield_strength = indicators["yield_strength"].to_numpy(dtype=float)
        ma200 = indicators["ma200"].to_numpy(dtype=float)

        n = len(close)
        raw_signal = np.zeros(n, dtype=np.int64)

        in_pos = 0
        entry_price = 0.0
        bars_in_pos = 0
        pt = float(params.profit_target_pct)
        ts = int(params.time_stop_bars)
        el_th = float(params.elastic_threshold)
        pl_th = float(params.plastic_threshold)

        for i in range(n):
            if in_pos != 0:
                bars_in_pos += 1
                exit_now = False
                if entry_price > 0.0 and np.isfinite(close[i]):
                    if in_pos > 0:
                        pnl = (close[i] - entry_price) / entry_price
                    else:
                        pnl = (entry_price - close[i]) / entry_price
                    if pnl >= pt:
                        exit_now = True
                if bars_in_pos >= ts:
                    exit_now = True
                if exit_now:
                    raw_signal[i] = 0
                    in_pos = 0
                    bars_in_pos = 0
                    entry_price = 0.0
                else:
                    raw_signal[i] = in_pos
                continue

            s = strain[i]
            a = atr[i]
            y = yield_strength[i]
            m = ma200[i]
            c = close[i]

            if not (
                np.isfinite(s)
                and np.isfinite(a)
                and np.isfinite(y)
                and np.isfinite(m)
                and np.isfinite(c)
            ):
                continue

            is_plastic = a > y

            new_sig = 0
            if is_plastic:
                if s >= pl_th and c > m:
                    new_sig = 1
                elif s <= -pl_th and c < m:
                    new_sig = -1
            else:
                if s >= el_th and c < m:
                    new_sig = -1
                elif s <= -el_th and c > m:
                    new_sig = 1

            if new_sig != 0:
                raw_signal[i] = new_sig
                in_pos = new_sig
                entry_price = c
                bars_in_pos = 0

        sig_series = pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        size_series = pd.Series(
            np.full(n, float(params.size_pct), dtype=float), index=data.index
        )

        df = pd.DataFrame({"signal": sig_series, "size": size_series}, index=data.index)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
