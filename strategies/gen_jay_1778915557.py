from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    acf_window: int = 30
    pred_window: int = 20
    atr_window: int = 14
    pred_decline_bars: int = 5
    breakeven_pct: float = 0.8
    trail_k: float = 2.0
    initial_stop_k: float = 2.5
    ma_window: int = 200
    base_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778915557"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(
            params.ma_window,
            params.acf_window + 2,
            params.pred_window + params.pred_decline_bars,
            params.atr_window + 1,
        ) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        gap = data["open"] / data["close"].shift(1) - 1
        ind["gap"] = gap

        # Vectorised lag-1 autocorrelation of gap via rolling cov / (std * std)
        gap_lag1 = gap.shift(1)
        cov = gap.rolling(params.acf_window).cov(gap_lag1)
        std_gap = gap.rolling(params.acf_window).std()
        std_lag = gap_lag1.rolling(params.acf_window).std()
        denom = std_gap * std_lag
        ind["gap_acf"] = (cov / denom).replace([np.inf, -np.inf], np.nan)

        # Predator density: rolling fraction of down-gaps (sellers as predators)
        neg_flag = (gap < 0).astype(float)
        ind["pred_density"] = neg_flag.rolling(params.pred_window).mean()

        # Predator retreat: negative slope means population collapsing
        ind["pred_slope"] = ind["pred_density"].diff(params.pred_decline_bars)

        # Confidence: ACF mean-reversion magnitude x predator retreat rate
        acf_reverts = (-ind["gap_acf"]).clip(lower=0)
        pred_retreats = (-ind["pred_slope"]).clip(lower=0)
        ind["confidence"] = acf_reverts * pred_retreats

        prev_close = data["close"].shift(1)
        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_window).mean()
        ind["ma200"] = data["close"].rolling(params.ma_window).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        closes = data["close"].values
        highs = data["high"].values
        atrs = indicators["atr"].values
        gap_acf = indicators["gap_acf"].values
        pred_slope = indicators["pred_slope"].values
        ma200 = indicators["ma200"].values
        gap_vals = indicators["gap"].values
        confidence = indicators["confidence"].values

        # Rolling 80th-pct of confidence for adaptive sizing normalisation
        conf_ref = (
            indicators["confidence"]
            .fillna(0)
            .rolling(60, min_periods=10)
            .quantile(0.80)
            .fillna(0)
            .values
        )

        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, params.base_size)

        in_position = False
        entry_price = 0.0
        stop_level = 0.0
        breakeven_activated = False
        entry_size = params.base_size

        for i in range(n):
            atr_i = atrs[i] if (not np.isnan(atrs[i]) and atrs[i] > 0) else 0.0

            if in_position:
                eff_atr = atr_i if atr_i > 0 else entry_price * 0.01
                high_i = highs[i]
                close_i = closes[i]

                if not breakeven_activated:
                    if high_i >= entry_price * (1.0 + params.breakeven_pct / 100.0):
                        breakeven_activated = True
                        stop_level = max(stop_level, entry_price)

                if breakeven_activated:
                    candidate = high_i - params.trail_k * eff_atr
                    if candidate > stop_level:
                        stop_level = candidate

                if close_i <= stop_level:
                    signals[i] = 0
                    in_position = False
                else:
                    signals[i] = 1
                    sizes[i] = entry_size

            else:
                if atr_i <= 0:
                    continue

                in_bull = (not np.isnan(ma200[i])) and (closes[i] > ma200[i])
                gap_down = gap_vals[i] < 0
                acf_neg = (not np.isnan(gap_acf[i])) and (gap_acf[i] < -0.05)
                pred_dec = (not np.isnan(pred_slope[i])) and (pred_slope[i] < 0)
                conf_i = 0.0 if np.isnan(confidence[i]) else confidence[i]
                has_conf = conf_i > 0.0

                if in_bull and gap_down and acf_neg and pred_dec and has_conf:
                    ref = conf_ref[i]
                    if np.isnan(ref) or ref <= 0.0:
                        ref = conf_i
                    norm = min(conf_i / ref, 1.0)
                    sz = params.base_size * (0.5 + 0.5 * norm)

                    signals[i] = 1
                    sizes[i] = sz
                    in_position = True
                    entry_price = closes[i]
                    stop_level = closes[i] - params.initial_stop_k * atr_i
                    breakeven_activated = False
                    entry_size = sz

        df = data.copy()
        df["signal"] = signals
        df["size"] = sizes
        # Mandatory 1-bar shift: decision on bar N close, fill on bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.base_size)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
