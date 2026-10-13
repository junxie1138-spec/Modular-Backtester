from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    drawdown_window: int = 63
    latent_window: int = 8
    noise_window: int = 20
    min_drawdown: float = 0.03
    innovation_threshold: float = 0.25
    hold_bars: int = 8

    def __post_init__(self):
        if min(self.drawdown_window, self.latent_window, self.noise_window) < 2:
            raise ValueError("Indicator windows must be at least two bars.")
        if self.hold_bars < 1:
            raise ValueError("hold_bars must be positive.")
        if not 0.0 < self.min_drawdown < 1.0:
            raise ValueError("min_drawdown must be between zero and one.")
        if not np.isfinite(self.innovation_threshold) or self.innovation_threshold < 0:
            raise ValueError("innovation_threshold must be finite and nonnegative.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791207296"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.drawdown_window, params.latent_window + 2,
                       params.noise_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_open"] = np.log(data["open"].where(data["open"] > 0))
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["previous_log_close"] = out["log_close"].shift(1)
        out["overnight"] = out["log_open"] - out["previous_log_close"]
        out["intraday"] = out["log_close"] - out["log_open"]
        out["total_return"] = out["log_close"].diff()
        # Each rolling mean estimates a separate latent session drift.
        out["overnight_slope"] = out["overnight"].rolling(params.latent_window).mean()
        out["intraday_slope"] = out["intraday"].rolling(params.latent_window).mean()
        out["net_slope"] = out["overnight_slope"] + out["intraday_slope"]
        out["overnight_innovation"] = out["overnight"] - out["overnight_slope"].shift(1)
        out["intraday_innovation"] = out["intraday"] - out["intraday_slope"].shift(1)
        out["noise"] = out["total_return"].rolling(params.noise_window).std(ddof=0)
        out["peak"] = data["close"].where(data["close"] > 0).rolling(params.drawdown_window).max()
        out["drawdown"] = data["close"].where(data["close"] > 0) / out["peak"] - 1.0
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        required = ["drawdown", "overnight_slope", "intraday_slope", "net_slope",
                    "overnight_innovation", "intraday_innovation", "noise"]
        valid = indicators[required].notna().all(axis=1)
        qualified = (
            valid
            & (indicators["drawdown"] <= -params.min_drawdown)
            & (indicators["overnight_slope"] < 0)
            & (indicators["intraday_slope"] > 0)
            & (indicators["net_slope"] > 0)
            & (indicators["overnight_innovation"] > 0)
            & (indicators["noise"] > 0)
            & (indicators["intraday_innovation"]
               > params.innovation_threshold * indicators["noise"])
        ).fillna(False)
        confirmed = qualified & qualified.shift(1, fill_value=False)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, enter in enumerate(confirmed.to_numpy(dtype=bool)):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    # Force the scheduled exit even when confirmation persists.
                    exit_at = -1
            elif enter:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params(hold_bars=3)
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({
        "drawdown": -0.05, "overnight_slope": -0.001,
        "intraday_slope": 0.002, "net_slope": 0.001,
        "overnight_innovation": 0.001, "intraday_innovation": 0.002,
        "noise": 0.001,
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 1, 1, 1, 0]
    indicators.loc[index[1], "intraday_innovation"] = np.nan
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].iloc[:4].eq(0).all()
    assert result["signal"].iloc[4:7].eq(1).all()
    assert result["signal"].iloc[7] == 0
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
