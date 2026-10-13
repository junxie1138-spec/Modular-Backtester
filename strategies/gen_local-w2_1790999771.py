from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecruitmentParams:
    volume_window: int = 20
    response_window: int = 12
    rank_window: int = 80
    atr_window: int = 14
    event_percentile: float = 0.80
    regime_percentile: float = 0.50
    trail_atr: float = 1.50
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RecruitmentParams]):
    strategy_id = "gen_local-w2_1790999771"

    @classmethod
    def params_type(cls) -> type[RecruitmentParams]:
        return RecruitmentParams

    def warmup_bars(self, params: RecruitmentParams) -> int:
        return int(max(
            params.volume_window + params.rank_window + 1,
            params.atr_window + params.rank_window + 1,
            params.response_window + params.rank_window + 3,
        ))

    def indicators(self, data: pd.DataFrame, params: RecruitmentParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        volume = data["volume"].astype(float).where(data["volume"] > 0)
        out = pd.DataFrame(index=data.index)
        out["return"] = close.pct_change(fill_method=None)
        out["absolute_return"] = out["return"].abs()
        out["log_volume_change"] = np.log(volume).diff()
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["volume_baseline"] = volume.rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean().shift(1)
        out["relative_volume"] = volume / out["volume_baseline"].replace(0.0, np.nan)
        out["impulse"] = close.diff() / out["atr"].replace(0.0, np.nan)
        out["impulse_magnitude"] = out["impulse"].abs()
        out["price_leads_volume"] = out["absolute_return"].shift(1).rolling(
            params.response_window, min_periods=params.response_window
        ).corr(out["log_volume_change"])
        out["volume_leads_price"] = out["log_volume_change"].shift(1).rolling(
            params.response_window, min_periods=params.response_window
        ).corr(out["absolute_return"])
        out["recruitment_balance"] = out["price_leads_volume"] - out["volume_leads_price"]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["volume_threshold"] = out["relative_volume"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).quantile(params.event_percentile).shift(1)
        out["impulse_threshold"] = out["impulse_magnitude"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).quantile(params.event_percentile).shift(1)
        out["regime_threshold"] = out["recruitment_balance"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).quantile(params.regime_percentile).shift(1)
        out["ready"] = out.notna().all(axis=1)
        out["continuation_regime"] = (
            out["recruitment_balance"] > out["regime_threshold"]
        ) & out["ready"]
        out["confirmed_event"] = (
            out["ready"]
            & (out["relative_volume"] > out["volume_threshold"])
            & (out["impulse_magnitude"] > out["impulse_threshold"])
            & (out["atr"] > 0)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecruitmentParams,
    ) -> SignalFrame:
        impulse_direction = np.sign(indicators["impulse"].fillna(0.0)).astype(int)
        regime_direction = pd.Series(
            np.where(indicators["continuation_regime"], 1, -1),
            index=data.index,
        )
        entries = (impulse_direction * regime_direction).where(
            indicators["confirmed_event"], 0
        ).astype(int).to_numpy()
        prices = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        highest_close = lowest_close = np.nan
        lower_stop = upper_stop = np.nan

        for i, price in enumerate(prices):
            if not (np.isfinite(price) and np.isfinite(atr[i]) and atr[i] > 0):
                position = 0
                age = 0
                continue
            if position:
                age += 1
                highest_close = max(highest_close, price)
                lowest_close = min(lowest_close, price)
                # The highest-close stop applies to both directions and only ratchets up.
                lower_stop = max(lower_stop, highest_close - params.trail_atr * atr[i])
                # Shorts additionally have a protective upside trail that only ratchets down.
                upper_stop = min(upper_stop, lowest_close + params.trail_atr * atr[i])
                if (
                    price <= lower_stop
                    or (position < 0 and price >= upper_stop)
                    or age >= params.hold_bars
                ):
                    position = 0
                    age = 0
            elif entries[i] != 0:
                position = int(entries[i])
                age = 0
                highest_close = lowest_close = price
                lower_stop = price - params.trail_atr * atr[i]
                upper_stop = price + params.trail_atr * atr[i]
            signals[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
