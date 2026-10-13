from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    drawdown_window: int = 20
    roc_window: int = 3
    atr_window: int = 14
    percentile_window: int = 126
    release_percentile: float = 0.85
    depth_percentile: float = 0.70
    stop_atr: float = 1.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791218333"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(
            params.drawdown_window + params.percentile_window + 1,
            params.roc_window + params.percentile_window + 3,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["previous_close"] = close.shift(1)
        out["peak"] = close.rolling(params.drawdown_window).max()
        out["depth"] = 1.0 - close / out["peak"]
        out["previous_depth"] = out["depth"].shift(1)
        out["roc"] = close.pct_change(params.roc_window, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["atr_fraction"] = out["atr"] / close
        out["release"] = (
            out["acceleration"] * out["previous_depth"]
            / out["atr_fraction"].where(out["atr_fraction"] > 0)
        )
        out["release_threshold"] = out["release"].rolling(
            params.percentile_window
        ).quantile(params.release_percentile).shift(1)
        out["depth_threshold"] = out["depth"].rolling(
            params.percentile_window
        ).quantile(params.depth_percentile).shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        out["entry"] = (
            ready
            & (out["previous_depth"] > out["depth_threshold"])
            & (out["previous_depth"] > 0)
            & (out["roc"] < 0)
            & (out["acceleration"] > 0)
            & (close > out["previous_close"])
            & (out["release"] > out["release_threshold"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        stop = np.nan
        held = 0
        for i in range(len(data)):
            if pending:
                active = bool(
                    np.isfinite(opens[i]) and opens[i] > 0
                    and np.isfinite(atr[i - 1]) and atr[i - 1] > 0
                )
                if active:
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                    held = 0
                pending = False
            exited = False
            if active:
                held += 1
                if closes[i] < stop or held >= params.hold_bars:
                    active = False
                    exited = True
            if not active and not exited and entries[i] and i + 1 < len(data):
                pending = True
            raw[i] = int(active or pending)
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100.0] * 7,
        "high": [101.0] * 7,
        "low": [96.0] * 7,
        "close": [100.0, 97.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 7,
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [True, False, False, True, False, False, False],
        "atr": [1.0] * 7,
    }, index=index)
    result = GeneratedStrategy().generate_signals(
        data, indicators, None, RecoveryParams()
    ).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 1, 1, 0]
    assert (result["size"] > 0).all()
