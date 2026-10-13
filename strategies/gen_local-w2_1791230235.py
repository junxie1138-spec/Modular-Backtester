from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    fine_share: float = 0.55
    polarity_threshold: float = 0.25


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791230235"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        # Eight returns for the coarse coefficient, then four energy bars.
        return 12

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if not (0.0 < params.fine_share < 1.0):
            raise ValueError("fine_share must lie strictly between zero and one")
        if not (0.0 < params.polarity_threshold < 1.0):
            raise ValueError("polarity_threshold must lie strictly between zero and one")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0)).replace([np.inf, -np.inf], np.nan)
        out["return"] = out["log_close"].diff()
        out["mean_2"] = out["return"].rolling(2, min_periods=2).mean()
        out["mean_4"] = out["return"].rolling(4, min_periods=4).mean()
        # Causal, undecimated Haar coefficients on adjacent return blocks.
        out["haar_1"] = (out["return"] - out["return"].shift(1)) / np.sqrt(2.0)
        out["haar_2"] = out["mean_2"] - out["mean_2"].shift(2)
        out["haar_4"] = np.sqrt(2.0) * (out["mean_4"] - out["mean_4"].shift(4))
        for scale in (1, 2, 4):
            out[f"power_{scale}"] = out[f"haar_{scale}"].pow(2)
            out[f"energy_{scale}"] = out[f"power_{scale}"].rolling(4, min_periods=4).mean()
        out["signed_power"] = (
            np.sign(out["haar_1"]) * out["power_1"]
            + np.sign(out["haar_2"]) * out["power_2"]
        )
        out["signed_energy"] = out["signed_power"].rolling(4, min_periods=4).mean()
        out["fine_energy"] = out["energy_1"] + out["energy_2"]
        out["total_energy"] = out["fine_energy"] + out["energy_4"]
        out["fine_share"] = out["fine_energy"] / out["total_energy"].replace(0.0, np.nan)
        out["polarity"] = out["signed_energy"] / out["fine_energy"].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        valid = indicators[["fine_share", "polarity"]].notna().all(axis=1)
        concentrated = valid & (indicators["fine_share"] >= params.fine_share)
        enter = concentrated & (indicators["polarity"] <= -params.polarity_threshold)
        reverse = concentrated & (indicators["polarity"] >= params.polarity_threshold)
        # Neutral energy states retain the position; opposite polarity exits.
        decisions = pd.Series(np.nan, index=data.index, dtype=float)
        decisions.loc[enter] = 1.0
        decisions.loc[reverse] = 0.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions.ffill().fillna(0.0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = WaveletParams()
    index = pd.date_range("2020-01-01", periods=240, freq="D")
    close = 100.0 * np.exp(np.random.default_rng(7).normal(0.0, 0.01, len(index)).cumsum())
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    features = strategy.indicators(data, params)
    result = strategy.generate_signals(data, features, ctx, params).data
    assert set(result["signal"].unique()) == {0, 1}
    assert (result["size"] > 0.0).all()
    assert not result["signal"].iloc[:12].any()
    assert (result["signal"].diff() == -1).any()
    prefix = data.iloc[:160]
    prefix_features = strategy.indicators(prefix, params)
    pd.testing.assert_frame_equal(features.iloc[:160], prefix_features)
    prefix_result = strategy.generate_signals(prefix, prefix_features, ctx, params).data
    pd.testing.assert_frame_equal(result.iloc[:160], prefix_result)
