from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeDecayParams:
    corr_window: int = 40
    atr_window: int = 14
    min_corr: float = 0.20
    min_elevation: float = 1.20
    min_contraction: float = 0.04
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RangeDecayParams]):
    strategy_id = "gen_local-w2_1791329503"

    @classmethod
    def params_type(cls):
        return RangeDecayParams

    def warmup_bars(self, params: RangeDecayParams) -> int:
        return int(max(200, params.corr_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RangeDecayParams) -> pd.DataFrame:
        if (params.corr_window < 3 or params.atr_window < 1
                or params.hold_bars < 1 or not 0 <= params.min_corr < 1
                or not np.isfinite(params.min_elevation) or params.min_elevation <= 1
                or not np.isfinite(params.min_contraction) or not 0 < params.min_contraction < 1
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0):
            raise ValueError("Invalid range-decay parameters")
        out = pd.DataFrame(index=data.index)
        high, low, close = data["high"], data["low"], data["close"]
        ratio = (high / low.where(low > 0)).where(high >= low)
        out["log_range"] = np.log(ratio).replace([np.inf, -np.inf], np.nan)
        out["lag_range"] = out["log_range"].shift(1)
        out["range_mean"] = out["log_range"].rolling(params.corr_window).mean()
        out["range_ac1"] = out["log_range"].rolling(params.corr_window).corr(
            out["lag_range"]
        ).clip(-1, 1)
        out["range_forecast"] = out["range_mean"] + out["range_ac1"] * (
            out["log_range"] - out["range_mean"]
        )
        out["forecast_contraction"] = (
            (out["log_range"] - out["range_forecast"])
            / out["log_range"].replace(0, np.nan)
        )
        out["prev_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            high - low,
            (high - out["prev_close"]).abs(),
            (low - out["prev_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["ma200"] = close.rolling(200).mean()
        out["close_location"] = (close - low) / (high - low).replace(0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeDecayParams,
    ) -> SignalFrame:
        required = ["log_range", "range_mean", "range_ac1",
                    "forecast_contraction", "atr", "ma200", "close_location"]
        ready = indicators[required].notna().all(axis=1)
        setup = (
            ready
            & (data["close"] > indicators["ma200"])
            & (indicators["atr"] > 0)
            & (indicators["range_ac1"] >= params.min_corr)
            & (indicators["log_range"] >= params.min_elevation * indicators["range_mean"])
            & (indicators["forecast_contraction"] >= params.min_contraction)
            & (indicators["close_location"] >= 0.70)
        ).fillna(False)
        entries = (setup & ~setup.shift(1, fill_value=False)).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        fixed_stop = np.nan
        for i in range(len(data)):
            if active:
                if entry_bar == -1:
                    # The prior decision fills at this open; freeze prior-close ATR.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        active = False
                        continue
                    entry_bar = i
                    fixed_stop = opens[i] - params.stop_atr * atr[i - 1]
                if (not np.isfinite(closes[i]) or closes[i] < fixed_stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    active = False
                    entry_bar = -1
                else:
                    signal[i] = 1
            elif entries[i]:
                active = True
                entry_bar = -1
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RangeDecayParams(hold_bars=3)
    strategy = GeneratedStrategy()
    idx = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": 100.0, "high": 102.0, "low": 94.0,
        "close": [100.0, 100.0, 99.0, 95.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=idx)
    ind = pd.DataFrame({
        "log_range": 0.02, "range_mean": 0.01, "range_ac1": 0.5,
        "forecast_contraction": 0.1, "atr": 2.0,
        "ma200": 90.0, "close_location": 0.5,
    }, index=idx)
    ind.loc[idx[1], "close_location"] = 0.8
    ind.loc[idx[2]:, "atr"] = 100.0
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    stopped = strategy.generate_signals(data, ind, ctx, params).data
    assert stopped["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0]
    data["close"] = 100.0
    expired = strategy.generate_signals(data, ind, ctx, params).data
    assert expired["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0]
    ind["ma200"] = 110.0
    assert not strategy.generate_signals(data, ind, ctx, params).data["signal"].any()
    flat = pd.DataFrame({
        "open": 100.0, "high": 100.0, "low": 100.0,
        "close": 100.0, "volume": 1000.0,
    }, index=pd.date_range("2020-01-01", periods=220))
    actual = strategy.indicators(flat, params)
    result = strategy.generate_signals(flat, actual, ctx, params).data
    assert actual.index.equals(flat.index)
    assert not result["signal"].any()
    assert (result["size"] > 0).all()
    assert strategy.warmup_bars(RangeDecayParams(corr_window=240)) == 241


if __name__ == "__main__":
    _self_check()
