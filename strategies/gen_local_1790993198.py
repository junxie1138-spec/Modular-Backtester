from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SponsorshipParams:
    channel_bars: int = 40
    sponsorship_bars: int = 10
    volume_bars: int = 20
    atr_bars: int = 14
    sponsorship_threshold: float = 0.03
    edge_threshold: float = 0.65
    volume_ratio: float = 1.05
    trail_atr: float = 3.0
    holding_bars: int = 20


class GeneratedStrategy(BaseStrategy[SponsorshipParams]):
    strategy_id = "gen_local_1790993198"

    @classmethod
    def params_type(cls) -> type[SponsorshipParams]:
        return SponsorshipParams

    @staticmethod
    def warmup_bars(params: SponsorshipParams) -> int:
        return int(max(params.channel_bars + 1,
                       params.sponsorship_bars,
                       params.volume_bars + 1,
                       params.atr_bars + 1) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SponsorshipParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        volume = data["volume"].clip(lower=0)
        out["prior_high"] = data["high"].rolling(params.channel_bars).max().shift(1)
        out["prior_low"] = data["low"].rolling(params.channel_bars).min().shift(1)
        out["channel_width"] = (out["prior_high"] - out["prior_low"]).replace(0.0, np.nan)
        out["relative_position"] = (close - out["prior_low"]) / out["channel_width"]
        out["volume_sum"] = volume.rolling(params.sponsorship_bars).sum()
        out["weighted_price_sum"] = (close * volume).rolling(params.sponsorship_bars).sum()
        out["volume_center"] = out["weighted_price_sum"] / out["volume_sum"].replace(0.0, np.nan)
        out["ordinary_center"] = close.rolling(params.sponsorship_bars).mean()
        out["sponsorship"] = (out["volume_center"] - out["ordinary_center"]) / out["channel_width"]
        out["baseline_volume"] = volume.rolling(params.volume_bars).mean().shift(1)
        out["relative_volume"] = volume / out["baseline_volume"].replace(0.0, np.nan)
        out["close_change"] = close.diff()
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SponsorshipParams) -> SignalFrame:
        required = ["relative_position", "sponsorship", "relative_volume", "close_change", "atr"]
        valid = indicators[required].notna().all(axis=1) & indicators["atr"].gt(0)
        active_volume = indicators["relative_volume"].ge(params.volume_ratio)
        long_setup = (
            valid & active_volume
            & indicators["relative_position"].ge(params.edge_threshold)
            & indicators["sponsorship"].ge(params.sponsorship_threshold)
            & indicators["close_change"].gt(0)
        )
        short_setup = (
            valid & active_volume
            & indicators["relative_position"].le(1.0 - params.edge_threshold)
            & indicators["sponsorship"].le(-params.sponsorship_threshold)
            & indicators["close_change"].lt(0)
        )
        long_confirmed = long_setup & long_setup.shift(1, fill_value=False)
        short_confirmed = short_setup & short_setup.shift(1, fill_value=False)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        longs = long_confirmed.to_numpy(dtype=bool)
        shorts = short_confirmed.to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = np.nan
        low_water = np.nan
        stop = np.nan

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            if position != 0:
                age += 1
                if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                    position = 0
                    continue
                # The first active close is on the bar after the entry decision.
                high_water = close if np.isnan(high_water) else max(high_water, close)
                low_water = close if np.isnan(low_water) else min(low_water, close)
                if position == 1:
                    candidate_stop = high_water - params.trail_atr * atr
                    stop = candidate_stop if np.isnan(stop) else max(stop, candidate_stop)
                    stopped = close <= stop
                else:
                    candidate_stop = low_water + params.trail_atr * atr
                    stop = candidate_stop if np.isnan(stop) else min(stop, candidate_stop)
                    stopped = close >= stop
                if stopped or age >= params.holding_bars:
                    position = 0
                    continue
            elif np.isfinite(close) and np.isfinite(atr) and atr > 0:
                if longs[i]:
                    position = 1
                elif shorts[i]:
                    position = -1
                if position != 0:
                    age = 0
                    high_water = np.nan
                    low_water = np.nan
                    stop = np.nan
            signals[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
