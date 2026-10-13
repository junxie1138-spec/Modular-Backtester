from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCouplingParams:
    coupling_window: int = 20
    atr_window: int = 14
    gap_atr: float = 0.35
    regime_threshold: float = 0.15
    stop_atr: float = 2.0
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[GapCouplingParams]):
    strategy_id = "gen_local-w2_1791255531"

    @classmethod
    def params_type(cls):
        return GapCouplingParams

    def warmup_bars(self, params: GapCouplingParams) -> int:
        return max(params.coupling_window, params.atr_window) + 3

    def indicators(self, data: pd.DataFrame, params: GapCouplingParams) -> pd.DataFrame:
        if params.coupling_window < 2 or params.atr_window < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 3 <= params.hold_days <= 5:
            raise ValueError("hold_days must be between three and five.")
        if not (np.isfinite(params.stop_atr) and params.stop_atr > 0
                and np.isfinite(params.gap_atr) and params.gap_atr > 0
                and np.isfinite(params.regime_threshold)
                and 0 < params.regime_threshold < 1):
            raise ValueError("Invalid gap, regime, or stop threshold.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        denominator = out["previous_close"].replace(0, np.nan)
        out["gap"] = (data["open"] - out["previous_close"]) / denominator
        out["body"] = (data["close"] - data["open"]) / denominator
        out["close_change"] = data["close"].diff()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["gap_atr"] = (
            (data["open"] - out["previous_close"])
            / out["atr"].shift(1).replace(0, np.nan)
        )
        out["cross_energy"] = (out["gap"] * out["body"]).rolling(params.coupling_window).mean()
        out["gap_energy"] = out["gap"].pow(2).rolling(params.coupling_window).mean()
        out["body_energy"] = out["body"].pow(2).rolling(params.coupling_window).mean()
        out["coupling_scale"] = np.sqrt(out["gap_energy"] * out["body_energy"])
        out["coupling"] = (
            out["cross_energy"] / out["coupling_scale"].replace(0, np.nan)
        ).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapCouplingParams,
    ) -> SignalFrame:
        values = indicators[["coupling", "gap_atr", "body", "close_change", "atr"]].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        targets = np.zeros(len(data), dtype=int)
        regime = 0
        previous_regime = 0
        previous_event = 0
        position = 0
        pending_entry = False
        entry_price = 0.0
        frozen_atr = 0.0
        age = 0
        for i in range(len(data)):
            coupling, gap, body, change, atr = values[i]
            valid = bool(np.isfinite(values[i]).all()
                         and np.isfinite(opens[i]) and opens[i] > 0
                         and np.isfinite(closes[i]) and closes[i] > 0 and atr > 0)
            if not valid:
                position = 0
                pending_entry = False
                previous_event = 0
                continue
            if coupling > params.regime_threshold:
                regime = 1
            elif coupling < -params.regime_threshold:
                regime = -1
            direction = int(np.sign(gap)) * regime
            event = direction if (
                abs(gap) >= params.gap_atr and direction != 0
                and direction * body > 0
            ) else 0
            confirmed = (
                previous_event != 0 and regime == previous_regime
                and previous_event * body > 0 and previous_event * change > 0
            )
            entry_direction = previous_event
            previous_event = event
            previous_regime = regime
            if position:
                if pending_entry:
                    # This open is observed only on the bar after the entry decision.
                    entry_price = opens[i]
                    pending_entry = False
                age += 1
                # Fixed entry ATR: long stop below entry, short stop above entry.
                stopped = position * (closes[i] - entry_price) < -params.stop_atr * frozen_atr
                if stopped or age >= params.hold_days:
                    position = 0
                targets[i] = position
                continue
            if confirmed:
                position = entry_direction
                frozen_atr = atr
                pending_entry = True
                age = 0
            targets[i] = position
        df = pd.DataFrame({"signal": targets, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapCouplingParams()
    for side in (1, -1):
        for stopped in (False, True):
            close = np.arange(101.0, 109.0)
            open_ = close - 0.5
            if stopped:
                open_[2] = 103.0
                close[2:] = 100.0
            close = 100.0 + side * (close - 100.0)
            open_ = 100.0 + side * (open_ - 100.0)
            index = pd.date_range("2020-01-01", periods=8)
            data = pd.DataFrame({
                "open": open_, "high": np.maximum(open_, close) + 1,
                "low": np.minimum(open_, close) - 1,
                "close": close, "volume": 1000.0,
            }, index=index)
            indicators = pd.DataFrame({
                "coupling": 1.0, "gap_atr": float(side),
                "body": 0.01 * side, "close_change": data["close"].diff().fillna(float(side)),
                "atr": 1.0,
            }, index=index)
            result = strategy.generate_signals(data, indicators, None, params).data
            assert result["signal"].iloc[:2].eq(0).all()
            assert result["signal"].iloc[2] == side
            if stopped:
                assert result["signal"].iloc[3] == 0
            else:
                assert result["signal"].iloc[2:6].eq(side).all()
                assert result["signal"].iloc[6] == 0
            assert result["size"].gt(0).all()
            derived = strategy.indicators(data, params)
            assert derived.index.equals(data.index)
            assert strategy.generate_signals(data, derived, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
