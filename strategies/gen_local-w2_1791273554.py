from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    area_window: int = 8
    baseline_window: int = 30
    fast_window: int = 5
    atr_window: int = 14
    area_threshold: float = 0.08
    expansion_ratio: float = 1.15
    breakeven_gain: float = 0.015
    trail_atr: float = 2.5
    initial_stop_atr: float = 2.5
    max_hold: int = 10
    cooldown_bars: int = 2


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791273554"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return int(max(params.baseline_window + params.area_window, params.fast_window + 1, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if min(params.area_window, params.baseline_window, params.fast_window, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two bars")
        if min(params.area_threshold, params.expansion_ratio, params.breakeven_gain, params.trail_atr, params.initial_stop_atr) <= 0:
            raise ValueError("Thresholds and stop distances must be positive")
        if params.max_hold < 1 or params.cooldown_bars < 0:
            raise ValueError("Invalid holding or cooldown duration")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        returns = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        volume_base = data["volume"].rolling(params.baseline_window, min_periods=params.baseline_window).mean().shift(1)
        surprise = (data["volume"] / volume_base.where(volume_base > 0) - 1.0).clip(-1.0, 3.0)
        dx = returns.fillna(0.0)
        dy = surprise.fillna(0.0)
        x = dx.cumsum()
        y = dy.cumsum()
        x_previous = x.shift(1).fillna(0.0)
        y_previous = y.shift(1).fillna(0.0)
        length = params.area_window
        x_origin = x.shift(length).fillna(0.0)
        y_origin = y.shift(length).fillna(0.0)
        cross = x_previous * dy - y_previous * dx
        sum_dx = dx.rolling(length, min_periods=length).sum()
        sum_dy = dy.rolling(length, min_periods=length).sum()
        area = 0.5 * (cross.rolling(length, min_periods=length).sum() - x_origin * sum_dy + y_origin * sum_dx)
        scale = dx.abs().rolling(length, min_periods=length).sum() * dy.abs().rolling(length, min_periods=length).sum()
        valid = (returns.notna() & surprise.notna()).rolling(length, min_periods=length).sum().eq(length)
        out["signature_area"] = (area / scale.where(scale > 0)).where(valid).replace([np.inf, -np.inf], np.nan)
        out["displacement"] = returns.rolling(length, min_periods=length).sum()
        out["volume_surprise"] = surprise
        slow_vol = returns.rolling(params.baseline_window, min_periods=params.baseline_window).std(ddof=0)
        fast_vol = returns.rolling(params.fast_window, min_periods=params.fast_window).std(ddof=0)
        out["volatility_ratio"] = (fast_vol / slow_vol.where(slow_vol > 0)).replace([np.inf, -np.inf], np.nan)
        previous_close = data["close"].shift(1)
        true_range = pd.concat([(data["high"] - data["low"]).abs(), (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["expansion"] = out["volatility_ratio"].ge(params.expansion_ratio)
        quiet_setup = ~out["expansion"] & out["signature_area"].le(-params.area_threshold)
        expansion_setup = out["expansion"] & out["signature_area"].ge(params.area_threshold)
        ready = out[["signature_area", "displacement", "volatility_ratio", "atr"]].notna().all(axis=1)
        out["entry_setup"] = (ready & out["atr"].gt(0) & out["displacement"].gt(0) & (quiet_setup | expansion_setup)).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        setup = indicators["entry_setup"].fillna(False).astype(bool)
        same_regime = indicators["expansion"].eq(indicators["expansion"].shift(1))
        confirmed = (setup & setup.shift(1, fill_value=False) & same_regime).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        armed = False
        entry = 0.0
        entry_atr = 0.0
        stop = 0.0
        peak = 0.0
        held = 0
        next_entry = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry = opens[i]
                    stop = entry - params.initial_stop_atr * entry_atr
                    peak = entry
                    armed = False
                    held = 0
                else:
                    next_entry = i + params.cooldown_bars + 1
                    continue
            if active:
                held += 1
                valid_bar = np.isfinite([opens[i], highs[i], lows[i], closes[i], atrs[i]]).all() and atrs[i] > 0
                exit_now = not valid_bar
                if valid_bar:
                    exit_now = opens[i] <= stop or lows[i] <= stop
                    if not exit_now:
                        peak = max(peak, highs[i])
                        if highs[i] >= entry * (1.0 + params.breakeven_gain):
                            armed = True
                        if armed:
                            stop = max(stop, entry, peak - params.trail_atr * atrs[i])
                            exit_now = closes[i] <= stop
                    exit_now = exit_now or held >= params.max_hold
                if exit_now:
                    active = False
                    next_entry = i + params.cooldown_bars + 1
                else:
                    signals[i] = 1
                continue
            if i >= next_entry and confirmed[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                signals[i] = 1
                pending = True
                entry_atr = atrs[i]
        df = pd.DataFrame({"signal": signals, "size": np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame({"open": [100.0] * 7, "high": [101.0, 101.0, 103.0, 102.0, 101.0, 101.0, 101.0], "low": [99.0, 99.0, 99.5, 99.0, 99.0, 99.0, 99.0], "close": [100.0, 100.0, 102.5, 100.0, 100.0, 100.0, 100.0], "volume": [1000.0] * 7}, index=index)
    features = pd.DataFrame({"entry_setup": [True, True, False, False, False, False, False], "expansion": [False] * 7, "atr": [1.0] * 7}, index=index)
    result = GeneratedStrategy.generate_signals(data, features, None, SignatureParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0]
    assert result["size"].gt(0).all()
    features["entry_setup"] = [True, False, True, False, True, False, True]
    result = GeneratedStrategy.generate_signals(data, features, None, SignatureParams()).data
    assert result["signal"].eq(0).all()
    assert GeneratedStrategy.indicators(data, SignatureParams())["entry_setup"].eq(False).all()
    assert GeneratedStrategy.warmup_bars(SignatureParams()) >= 38


if __name__ == "__main__":
    _self_check()
