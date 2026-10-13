from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    regime_bars: int = 20
    volume_bars: int = 20
    breakout_bars: int = 20
    recovery_bars: int = 5
    atr_bars: int = 14
    stress_share: float = 0.65
    volume_multiple: float = 1.3
    move_atr: float = 0.35
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791247214"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.regime_bars, params.volume_bars,
                   params.breakout_bars, params.recovery_bars,
                   params.atr_bars) + 1

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["return_power"] = out["return"].pow(2)
        out["downside_power"] = out["return"].clip(upper=0).pow(2)
        out["total_power"] = out["return_power"].rolling(params.regime_bars).sum()
        out["down_power"] = out["downside_power"].rolling(params.regime_bars).sum()
        out["downside_share"] = out["down_power"].div(out["total_power"].replace(0, np.nan))
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["baseline_volume"] = data["volume"].rolling(params.volume_bars).mean().shift(1)
        out["relative_volume"] = data["volume"].div(out["baseline_volume"].replace(0, np.nan))
        out["body_atr"] = (data["close"] - data["open"]).div(out["atr"].replace(0, np.nan))
        out["breakout_level"] = data["high"].rolling(params.breakout_bars).max().shift(1)
        out["recovery_level"] = data["high"].rolling(params.recovery_bars).max().shift(1)
        out["stress"] = out["downside_share"] >= params.stress_share
        out["entry_level"] = out["breakout_level"].where(~out["stress"], out["recovery_level"])
        out["volume_move"] = (
            (out["relative_volume"] >= params.volume_multiple)
            & (out["body_atr"] >= params.move_atr)
            & (data["close"] > out["previous_close"])
        )
        out["price_acceptance"] = data["close"] > out["entry_level"]
        ready = np.isfinite(out[["downside_share", "relative_volume", "body_atr",
                                "entry_level", "atr"]]).all(axis=1)
        out["entry"] = ready & (out["atr"] > 0) & out["volume_move"] & out["price_acceptance"]
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        if params.stop_atr <= 0 or params.hold_bars < 1:
            raise ValueError("stop_atr and hold_bars must be positive")
        raw = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        stop = np.nan
        entry_atr = np.nan
        held = 0
        for i in range(len(data)):
            if pending:
                # The order decided yesterday fills at today's open.
                stop = float(data["open"].iloc[i]) - params.stop_atr * entry_atr
                pending = False
                held = 0
            if active:
                held += 1
                close = float(data["close"].iloc[i])
                if not np.isfinite(close) or not np.isfinite(stop) or close < stop or held >= params.hold_bars:
                    active = False
                else:
                    raw[i] = 1
            elif bool(indicators["entry"].iloc[i]):
                entry_atr = float(indicators["atr"].iloc[i])
                if np.isfinite(entry_atr) and entry_atr > 0:
                    active = True
                    pending = True
                    raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=24)
    data = pd.DataFrame({"open": 100.0, "high": 111.0, "low": 95.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    fixture = pd.DataFrame({"entry": False, "atr": 1.0}, index=index)
    fixture.iloc[0, 0] = True
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    data.iloc[1, data.columns.get_loc("close")] = 110.0
    data.iloc[3, data.columns.get_loc("close")] = 96.0
    stopped = strategy.generate_signals(data, fixture, ctx, params).data["signal"]
    assert stopped.iloc[:5].tolist() == [0, 1, 1, 1, 0]
    data["close"] = 100.0
    timed = strategy.generate_signals(data, fixture, ctx, params).data["signal"]
    assert timed.tolist() == [0] + [1] * 18 + [0] * 5
