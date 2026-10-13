from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OutrunParams:
    ma_window: int = 20
    atr_window: int = 14
    z_entry: float = 0.6
    z_ceiling: float = 2.0
    outrun_ratio: float = 2.0
    breakeven_pct: float = 0.004
    trail_atr: float = 0.9
    initial_stop_atr: float = 1.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[OutrunParams]):
    strategy_id = "gen_local-w2_1790972558"

    @classmethod
    def params_type(cls):
        return OutrunParams

    def warmup_bars(self, params: OutrunParams) -> int:
        return int(max(params.ma_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: OutrunParams) -> pd.DataFrame:
        if (params.ma_window < 2 or params.atr_window < 2
                or not 0 < params.z_entry < params.z_ceiling
                or params.outrun_ratio <= 0 or params.breakeven_pct <= 0
                or params.trail_atr <= 0 or params.initial_stop_atr <= 0
                or params.hold_bars not in (1, 2)):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window).std(ddof=0).replace(0, np.nan)
        out["z"] = (close - out["ma"]) / out["std"]
        out["price_step"] = close.diff()
        out["ma_step"] = out["ma"].diff()
        out["z_step"] = out["z"].diff()
        previous_close = close.shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["tr"].rolling(params.atr_window).mean().replace(0, np.nan)
        valid = np.isfinite(out[["z", "price_step", "ma_step", "z_step", "atr"]]).all(axis=1)
        outruns = out["price_step"].abs() > params.outrun_ratio * out["ma_step"].abs()
        moderate = out["z"].abs().between(params.z_entry, params.z_ceiling)
        long_ok = valid & outruns & moderate & (out["z"] > 0) & (out["price_step"] > 0) & (out["z_step"] > 0)
        short_ok = valid & outruns & moderate & (out["z"] < 0) & (out["price_step"] < 0) & (out["z_step"] < 0)
        out["candidate"] = np.select([long_ok, short_ok], [1, -1], default=0).astype(int)
        out["confirmed"] = out["candidate"].where(
            out["candidate"].eq(out["candidate"].shift(1)), 0
        ).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: OutrunParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        direction = 0
        entry_signed = np.nan
        stop_signed = np.nan
        armed = False
        age = 0
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["confirmed"].to_numpy(dtype=int)
        for i in range(len(data)):
            if direction:
                # A decision on the preceding close fills at this bar's open.
                if age == 0:
                    if not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(atrs[i - 1]):
                        direction = 0
                        continue
                    entry_signed = direction * opens[i]
                    stop_signed = entry_signed - params.initial_stop_atr * atrs[i - 1]
                age += 1
                if not np.isfinite(closes[i]) or not np.isfinite(atrs[i]):
                    direction = 0
                    continue
                marked = direction * closes[i]
                # Close-triggered exits execute at the following open.
                if marked <= stop_signed or age >= params.hold_bars:
                    direction = 0
                    continue
                if marked - entry_signed >= abs(entry_signed) * params.breakeven_pct:
                    armed = True
                if armed:
                    # Signed stops increase for both sides: long stops rise,
                    # short buy-to-cover stops fall, never loosening risk.
                    stop_signed = max(stop_signed, entry_signed, marked - params.trail_atr * atrs[i])
                decisions[i] = direction
            elif entries[i] and np.isfinite(atrs[i]) and np.isfinite(closes[i]) and closes[i] > 0:
                direction = int(entries[i])
                age = 0
                armed = False
                decisions[i] = direction
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = OutrunParams()
    index = pd.date_range("2020-01-01", periods=80)
    close = 100 + np.sin(np.arange(80) * 0.4) + np.arange(80) * 0.03
    data = pd.DataFrame({"open": close, "high": close + 0.5,
                         "low": close - 0.5, "close": close, "volume": 1000.0}, index=index)
    ind = strategy.indicators(data, params)
    active = ind["confirmed"].ne(0)
    assert active.any()
    assert (ind.loc[active, "candidate"] == ind["candidate"].shift(1).loc[active]).all()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    prefix = strategy.generate_signals(data.iloc[:60], strategy.indicators(data.iloc[:60], params), ctx, params).data
    assert result.iloc[:60].equals(prefix)
    assert result["signal"].isin([-1, 0, 1]).all() and result["size"].gt(0).all()
    # Mirrored paths check execution delay, stop breach and two-session cap.
    for side in (1, -1):
        sample = data.iloc[:5].copy()
        sample["open"] = 100.0
        sample["close"] = [100.0, 100.0 - side * 2.0, 100.0, 100.0, 100.0]
        manual = pd.DataFrame({"atr": 1.0, "confirmed": [side, 0, 0, 0, 0]}, index=sample.index)
        stopped = strategy.generate_signals(sample, manual, ctx, params).data["signal"].tolist()
        assert stopped == [0, side, 0, 0, 0]
        sample["close"] = [100.0, 100.0 + side * 0.5, 100.0, 100.0, 100.0]
        held = strategy.generate_signals(sample, manual, ctx, params).data["signal"].tolist()
        assert held == [0, side, side, 0, 0]


if __name__ == "__main__":
    _self_check()
