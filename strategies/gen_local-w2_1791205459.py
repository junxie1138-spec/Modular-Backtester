from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 60
    trough_window: int = 20
    volume_window: int = 5
    floor_window: int = 3
    atr_window: int = 14
    minimum_drawdown: float = 0.04
    remaining_drawdown: float = 0.01
    advancing_volume_share: float = 0.60
    floor_lift_atr: float = 0.25
    stop_atr: float = 2.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791205459"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.peak_window + 1, params.trough_window,
                       params.volume_window + 1, 2 * params.floor_window,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        windows = (params.peak_window, params.trough_window,
                   params.volume_window, params.floor_window,
                   params.atr_window, params.holding_bars)
        if any(window < 1 for window in windows):
            raise ValueError("Windows and holding_bars must be positive")
        if not (0 < params.advancing_volume_share < 1):
            raise ValueError("advancing_volume_share must be between zero and one")
        if not (params.stop_atr > 0 and params.floor_lift_atr >= 0
                and 0 <= params.remaining_drawdown < params.minimum_drawdown < 1):
            raise ValueError("Invalid stop, floor lift, or drawdown thresholds")

        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous_close = close.shift(1)
        out["close_change"] = close.diff()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["prior_peak"] = close.rolling(params.peak_window).max().shift(1)
        out["trough"] = data["low"].rolling(params.trough_window).min()
        out["scar_depth"] = 1.0 - out["trough"] / out["prior_peak"].replace(0, np.nan)
        out["current_drawdown"] = 1.0 - close / out["prior_peak"].replace(0, np.nan)

        volume = data["volume"].where(data["volume"] >= 0)
        out["advancing_volume"] = volume.where(out["close_change"] > 0, 0.0).where(out["close_change"].notna())
        out["advancing_volume_sum"] = out["advancing_volume"].rolling(params.volume_window).sum()
        out["total_volume_sum"] = volume.rolling(params.volume_window).sum()
        out["advancing_share"] = out["advancing_volume_sum"] / out["total_volume_sum"].replace(0, np.nan)
        out["recent_floor"] = data["low"].rolling(params.floor_window).min()
        out["previous_floor"] = out["recent_floor"].shift(params.floor_window)
        out["floor_lift"] = (out["recent_floor"] - out["previous_floor"]) / out["atr"].replace(0, np.nan)

        out["volume_agrees"] = (
            (out["advancing_share"] >= params.advancing_volume_share)
            & (out["close_change"] > 0)
        )
        out["floor_agrees"] = (
            (out["floor_lift"] >= params.floor_lift_atr)
            & (out["recent_floor"] > out["trough"])
        )
        required = ["atr", "prior_peak", "trough", "advancing_share", "floor_lift"]
        ready = np.isfinite(out[required]).all(axis=1) & (out["atr"] > 0)
        out["candidate"] = (
            ready
            & (out["scar_depth"] >= params.minimum_drawdown)
            & (out["current_drawdown"] >= params.remaining_drawdown)
            & out["volume_agrees"]
            & out["floor_agrees"]
        ).fillna(False)
        out["entry_event"] = out["candidate"] & ~out["candidate"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_event"].fillna(False).to_numpy(dtype=bool)
        pending_entry = False
        entry_bar = -1
        fixed_stop = np.nan

        for i in range(len(data)):
            if pending_entry:
                # The prior close's decision fills at this bar's open.
                entry_bar = i
                fixed_stop = opens[i] - params.stop_atr * atr[i - 1]
                pending_entry = False

            if entry_bar >= 0:
                stopped = closes[i] < fixed_stop
                expired = i - entry_bar + 1 >= params.holding_bars
                if stopped or expired:
                    entry_bar = -1
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                decisions[i] = 1
                pending_entry = True

        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [120.0, 100.0, 99.0, 97.0, 97.0],
        "high": [121.0, 101.0, 100.0, 98.0, 98.0],
        "low": [119.0, 98.0, 96.0, 96.0, 96.0],
        "close": [120.0, 99.0, 97.0, 97.0, 97.0],
        "volume": [100.0] * 5,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0] * 5,
        "entry_event": [True, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, RecoveryParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0]
    timed = strategy.generate_signals(
        data, indicators, None, RecoveryParams(stop_atr=100.0, holding_bars=2)
    )
    assert timed.data["signal"].tolist() == [0, 1, 1, 0, 0]
    assert (result.data["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
