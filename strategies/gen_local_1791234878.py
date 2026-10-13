from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockParams:
    isolation_ratio: float = 0.25
    shock_share: float = 0.55
    residual_floor: float = 0.55
    shock_floor: float = 0.012
    profit_target: float = 0.03
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[ShockParams]):
    strategy_id = "gen_local_1791234878"

    @classmethod
    def params_type(cls) -> type[ShockParams]:
        return ShockParams

    def warmup_bars(self, params: ShockParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: ShockParams) -> pd.DataFrame:
        if not (0 < params.isolation_ratio <= 1 and 0 < params.shock_share <= 1
                and 0 < params.residual_floor <= 1.15 and params.shock_floor > 0
                and params.profit_target > 0 and params.max_hold_bars > 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["older_max"] = out["return"].shift(4).rolling(4, min_periods=4).max()
        out["older_min"] = out["return"].shift(4).rolling(4, min_periods=4).min()
        out["shock"] = out["older_max"].where(
            out["older_max"].abs() >= out["older_min"].abs(), out["older_min"]
        )
        out["shock_abs"] = out["shock"].abs()
        out["net_return"] = out["return"].rolling(8, min_periods=8).sum()
        out["absolute_load"] = out["return"].abs().rolling(8, min_periods=8).sum()
        out["recent_energy"] = out["return"].abs().rolling(4, min_periods=4).mean()
        out["residual"] = out["net_return"] / out["shock"].replace(0, np.nan)
        out["concentration"] = out["shock_abs"] / out["absolute_load"].replace(0, np.nan)
        out["quiet_ratio"] = out["recent_energy"] / out["shock_abs"].replace(0, np.nan)
        out["eligible"] = (
            (out["shock_abs"] >= params.shock_floor)
            & (out["concentration"] >= params.shock_share)
            & (out["quiet_ratio"] <= params.isolation_ratio)
            & out["residual"].between(params.residual_floor, 1.15)
        ).fillna(False)
        out["entry"] = np.where(out["eligible"], -np.sign(out["shock"].fillna(0)), 0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ShockParams,
    ) -> SignalFrame:
        if params.profit_target <= 0 or params.max_hold_bars <= 0:
            raise ValueError("Exit parameters must be positive")
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        position = 0
        entry_price = 0.0
        entry_bar = -1
        for i in range(len(data)):
            if position:
                if entry_bar < 0:
                    # The previous close's decision fills at this bar's open.
                    entry_price = opens[i]
                    entry_bar = i
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                        entry_bar = -1
                        continue
                gain = position * (closes[i] / entry_price - 1.0)
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if expired or (np.isfinite(gain) and gain >= params.profit_target):
                    position = 0
                    entry_bar = -1
                    continue
            elif entries[i] in (-1, 1) and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_bar = -1
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=5)
    for side in (1, -1):
        for target_hit in (False, True):
            close = [100.0] * 5
            if target_hit:
                close[1] = 104.0 if side == 1 else 96.0
            data = pd.DataFrame({"open": 100.0, "high": 105.0, "low": 95.0,
                                 "close": close, "volume": 1000.0}, index=index)
            entries = pd.DataFrame({"entry": [side, 0, 0, 0, 0]}, index=index)
            result = strategy.generate_signals(
                data, entries, None, ShockParams(max_hold_bars=3)
            ).data
            expected = [0, side, 0, 0, 0] if target_hit else [0, side, side, side, 0]
            assert result["signal"].tolist() == expected
            assert (result["size"] > 0).all()
    flat = data.assign(close=100.0)
    derived = strategy.indicators(flat, ShockParams())
    assert derived.index.equals(flat.index)
    assert derived["entry"].eq(0).all()
    assert strategy.warmup_bars(ShockParams()) == 9


if __name__ == "__main__":
    _self_check()
