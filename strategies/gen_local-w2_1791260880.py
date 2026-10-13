from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureBreakoutParams:
    signature_window: int = 12
    volume_window: int = 20
    breakout_window: int = 20
    area_threshold: float = 0.12
    atr_window: int = 14
    trail_atr: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[SignatureBreakoutParams]):
    strategy_id = "gen_local-w2_1791260880"

    @classmethod
    def params_type(cls) -> type[SignatureBreakoutParams]:
        return SignatureBreakoutParams

    def warmup_bars(self, params: SignatureBreakoutParams) -> int:
        return int(max(200, params.volume_window + params.signature_window + 1,
                       params.breakout_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: SignatureBreakoutParams) -> pd.DataFrame:
        windows = (params.signature_window, params.volume_window,
                   params.breakout_window, params.atr_window, params.holding_bars)
        if any(w < 1 for w in windows):
            raise ValueError("Lookbacks and holding_bars must be positive")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite")
        if not np.isfinite(params.area_threshold) or not 0 <= params.area_threshold <= 1:
            raise ValueError("area_threshold must lie between zero and one")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] >= 0)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["log_return"] = np.log(close).diff()
        out["volume_mean"] = volume.rolling(params.volume_window, min_periods=params.volume_window).mean()
        out["relative_volume"] = (volume / out["volume_mean"].where(out["volume_mean"] > 0)).clip(upper=5.0)

        # X is cumulative participation; Y is cumulative log-price movement.
        dx = out["relative_volume"].replace([np.inf, -np.inf], np.nan)
        dy = out["log_return"].replace([np.inf, -np.inf], np.nan)
        out["path_x"] = dx.fillna(0.0).cumsum()
        out["path_y"] = dy.fillna(0.0).cumsum()
        out["cross_increment"] = out["path_x"].shift(1) * dy - out["path_y"].shift(1) * dx
        length = params.signature_window
        out["participation_sum"] = dx.rolling(length, min_periods=length).sum()
        out["return_sum"] = dy.rolling(length, min_periods=length).sum()
        out["absolute_return_sum"] = dy.abs().rolling(length, min_periods=length).sum()
        out["signature_area"] = (
            out["cross_increment"].rolling(length, min_periods=length).sum()
            - out["path_x"].shift(length) * out["return_sum"]
            + out["path_y"].shift(length) * out["participation_sum"]
        )
        denominator = out["participation_sum"] * out["absolute_return_sum"]
        out["normalized_area"] = (out["signature_area"] / denominator.where(denominator > 0)).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        out["channel_high"] = data["high"].shift(1).rolling(params.breakout_window, min_periods=params.breakout_window).max()
        out["channel_low"] = data["low"].shift(1).rolling(params.breakout_window, min_periods=params.breakout_window).min()
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureBreakoutParams) -> SignalFrame:
        required = ["ma200", "normalized_area", "channel_high", "channel_low", "atr"]
        valid = pd.Series(np.isfinite(indicators[required].to_numpy()).all(axis=1), index=data.index)
        valid &= np.isfinite(data["close"]) & (data["close"] > 0) & (indicators["atr"] > 0)
        long_entry = (valid & (data["close"] > indicators["ma200"])
                      & (data["close"] > indicators["channel_high"])
                      & (indicators["normalized_area"] > params.area_threshold))
        short_entry = (valid & (data["close"] < indicators["ma200"])
                       & (data["close"] < indicators["channel_low"])
                       & (indicators["normalized_area"] < -params.area_threshold))
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        long_flags = long_entry.to_numpy(dtype=bool)
        short_flags = short_entry.to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = 0
        watermark = 0.0
        stop = 0.0

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
                position = 0
                continue
            if position != 0:
                if position == 1:
                    watermark = max(watermark, close)
                    stop = max(stop, watermark - params.trail_atr * atr)
                    stopped = close <= stop
                else:
                    # Mirror the high-water trailing stop for short positions.
                    watermark = min(watermark, close)
                    stop = min(stop, watermark + params.trail_atr * atr)
                    stopped = close >= stop
                if stopped or i - entry_bar >= params.holding_bars:
                    position = 0
                    continue
            elif long_flags[i] or short_flags[i]:
                position = 1 if long_flags[i] else -1
                entry_bar = i
                watermark = close
                stop = close - position * params.trail_atr * atr
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        assert df["signal"].isin([-1, 0, 1]).all() and (df["size"] > 0).all()
        return SignalFrame(data=df, signal_column="signal", size_column="size")
