from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FlowExhaustionParams:
    volume_window: int = 20
    flow_window: int = 3
    rank_window: int = 60
    tail_percentile: float = 0.15
    impact_percentile: float = 0.30
    atr_window: int = 14
    stop_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[FlowExhaustionParams]):
    strategy_id = "gen_local-w2_1791277703"

    @classmethod
    def params_type(cls):
        return FlowExhaustionParams

    def warmup_bars(self, params: FlowExhaustionParams) -> int:
        return int(max(params.volume_window, params.atr_window) + params.flow_window + params.rank_window + 1)

    def indicators(self, data: pd.DataFrame, params: FlowExhaustionParams) -> pd.DataFrame:
        if (min(params.volume_window, params.flow_window, params.rank_window, params.atr_window) < 2
                or not 0.0 < params.tail_percentile < 0.5
                or not 0.0 < params.impact_percentile < 1.0
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0
                or not 3 <= params.hold_bars <= 5):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        volume = data["volume"].where(data["volume"] >= 0.0)
        out["return"] = close.div(close.shift(1)).sub(1.0)
        out["volume_mean"] = volume.rolling(params.volume_window).mean()
        out["relative_volume"] = volume.div(out["volume_mean"].where(out["volume_mean"] > 0.0))
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["signed_volume"] = np.sign(out["return"]) * out["relative_volume"]
        out["flow"] = out["signed_volume"].rolling(params.flow_window).sum()
        out["effort"] = out["relative_volume"].rolling(params.flow_window).sum()
        out["net_return"] = close.div(close.shift(params.flow_window)).sub(1.0)
        out["impact"] = out["net_return"].abs().div(
            out["effort"].where(out["effort"] > 0.0)
            * out["atr"].where(out["atr"] > 0.0).div(close)
        )
        out["flow_percentile"] = out["flow"].rolling(params.rank_window).rank(pct=True)
        out["impact_percentile"] = out["impact"].rolling(params.rank_window).rank(pct=True)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: FlowExhaustionParams,
    ) -> SignalFrame:
        ready = (
            indicators[["flow_percentile", "impact_percentile", "net_return", "atr"]].notna().all(axis=1)
            & indicators["atr"].gt(0.0)
            & indicators["impact_percentile"].le(params.impact_percentile)
        )
        long_entry = ready & indicators["flow_percentile"].le(params.tail_percentile) & indicators["net_return"].lt(0.0)
        short_entry = ready & indicators["flow_percentile"].ge(1.0 - params.tail_percentile) & indicators["net_return"].gt(0.0)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if position == 0 and i > 0 and targets[i - 1] != 0:
                if np.isfinite(opens[i]) and opens[i] > 0.0 and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0.0:
                    position = int(targets[i - 1])
                    entry_bar = i
                    # Fill price is this bar's open; ATR was known at the previous close.
                    stop = opens[i] - position * params.stop_atr * atrs[i - 1]
            if position != 0:
                stopped = not np.isfinite(closes[i]) or position * (closes[i] - stop) < 0.0
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    position = 0
                    continue
                targets[i] = position
            elif np.isfinite(closes[i]) and closes[i] > 0.0:
                targets[i] = candidates[i]
        df = pd.DataFrame({"signal": targets, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = object.__new__(GeneratedStrategy)
    params = FlowExhaustionParams()
    index = pd.date_range("2020-01-01", periods=15)
    data = pd.DataFrame({
        "open": 100.0, "high": 105.0, "low": 95.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    data.loc[index[2], "close"] = 96.0
    data.loc[index[6], "close"] = 104.0
    ind = pd.DataFrame({
        "flow_percentile": 0.5, "impact_percentile": 0.5,
        "net_return": 0.0, "atr": 100.0,
    }, index=index)
    for bar, percentile, direction in [(0, 0.1, -1.0), (4, 0.9, 1.0), (8, 0.1, -1.0)]:
        ind.loc[index[bar], ["flow_percentile", "impact_percentile", "net_return", "atr"]] = [percentile, 0.1, direction, 2.0]
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 1, 1, 0, 0]
    assert result["size"].gt(0.0).all()
    actual = strategy.indicators(data, params)
    assert strategy.generate_signals(data, actual, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
