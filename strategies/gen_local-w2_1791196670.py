from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PlateauParams:
    z_jump: float = 1.0
    trail_atr: float = 1.2


class GeneratedStrategy(BaseStrategy[PlateauParams]):
    strategy_id = "gen_local-w2_1791196670"

    @classmethod
    def params_type(cls) -> type[PlateauParams]:
        return PlateauParams

    @staticmethod
    def warmup_bars(params: PlateauParams) -> int:
        # Twenty closes, followed by five prior standardized readings.
        return 25

    @staticmethod
    def indicators(data: pd.DataFrame, params: PlateauParams) -> pd.DataFrame:
        if not np.isfinite(params.z_jump) or params.z_jump <= 0:
            raise ValueError("z_jump must be finite and positive")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(20, min_periods=20).mean()
        out["dispersion"] = close.rolling(20, min_periods=20).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["dispersion"].replace(0.0, np.nan)
        prior_z = out["z"].shift(1)
        out["plateau"] = prior_z.rolling(5, min_periods=5).mean()
        out["plateau_dispersion"] = prior_z.rolling(5, min_periods=5).std(ddof=0)
        out["jump"] = out["z"] - out["plateau"]
        prior_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prior_close).abs(),
            (data["low"] - prior_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        ready = np.isfinite(out[["z", "plateau_dispersion", "jump", "atr"]]).all(axis=1)
        trigger = (
            ready
            & out["atr"].gt(0)
            & out["plateau_dispersion"].le(0.55)
            & out["jump"].abs().ge(params.z_jump)
            & out["z"].abs().ge(1.0)
            & (out["z"] * out["jump"]).gt(0)
        )
        out["entry_direction"] = np.select([
            trigger & out["jump"].gt(0),
            trigger & out["jump"].lt(0),
        ], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PlateauParams,
    ) -> SignalFrame:
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        pending = 0
        age = 0
        entry = 0.0
        stop = 0.0
        armed = False

        for i in range(len(data)):
            if pending:
                # Yesterday's decision fills at today's open.
                if i == 0 or not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(atrs[i - 1]) or atrs[i - 1] <= 0:
                    pending = 0
                    continue
                position = pending
                pending = 0
                entry = opens[i]
                # Signed prices let the protective floor only increase
                # for either direction; a short's actual price stop falls.
                stop = position * entry - 1.5 * atrs[i - 1]
                armed = False
                age = 0

            if position:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0 or not np.isfinite(atrs[i]) or atrs[i] <= 0:
                    position = 0
                    continue
                signed_close = position * closes[i]
                if position * (closes[i] / entry - 1.0) >= 0.005:
                    armed = True
                if armed:
                    stop = max(stop, position * entry, signed_close - params.trail_atr * atrs[i])
                # Stops are evaluated at the close and exit next open.
                if signed_close <= stop or age >= 2:
                    position = 0
                signals[i] = position
                continue

            if entries[i] in (-1, 1) and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atrs[i]) and atrs[i] > 0:
                pending = int(entries[i])
                signals[i] = pending

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    params = PlateauParams()
    for direction in (1, -1):
        derived = pd.DataFrame({"atr": 1.0, "entry_direction": [direction, 0, 0, 0, 0]}, index=index)
        result = GeneratedStrategy.generate_signals(data, derived, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, 0, 0]
        adverse = data.copy()
        adverse.loc[index[1], "close"] = 100.0 - direction * 2.0
        stopped = GeneratedStrategy.generate_signals(adverse, derived, None, params).data
        assert stopped["signal"].tolist() == [0, direction, 0, 0, 0]
        assert result["size"].gt(0).all()
    flat = GeneratedStrategy.indicators(data, params)
    assert flat["entry_direction"].eq(0).all()
