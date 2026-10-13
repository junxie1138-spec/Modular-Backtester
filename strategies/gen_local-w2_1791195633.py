from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    correlation_window: int = 24
    percentile_window: int = 126
    entry_percentile: float = 0.85
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791195633"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return params.correlation_window + params.percentile_window + 17

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if params.correlation_window < 3 or params.percentile_window < 3:
            raise ValueError("Rolling windows must be at least three bars.")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must lie strictly between zero and one.")
        out = pd.DataFrame(index=data.index)
        out["return"] = np.log(data["close"].where(data["close"] > 0)).diff()
        energies = []
        contributions = []
        for half in (1, 2, 4):
            tag = str(2 * half)
            out["block_" + tag] = out["return"].rolling(half).sum()
            out["detail_" + tag] = (
                out["block_" + tag] - out["block_" + tag].shift(half)
            ) / np.sqrt(2 * half)
            detail = out["detail_" + tag]
            out["lag_" + tag] = detail.shift(2 * half)
            out["energy_" + tag] = detail.pow(2).rolling(params.correlation_window).mean()
            out["rho_" + tag] = detail.rolling(params.correlation_window).corr(out["lag_" + tag]).clip(-1.0, 1.0)
            out["contribution_" + tag] = out["rho_" + tag] * detail * np.sqrt(out["energy_" + tag])
            energies.append("energy_" + tag)
            contributions.append("contribution_" + tag)
        out["total_energy"] = out[energies].sum(axis=1, min_count=3)
        out["score"] = (
            out[contributions].sum(axis=1, min_count=3)
            / out["total_energy"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        out["strength"] = out["score"].abs()
        out["threshold"] = out["strength"].shift(1).rolling(
            params.percentile_window
        ).quantile(params.entry_percentile)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        if params.holding_bars not in (1, 2):
            raise ValueError("holding_bars must be one or two.")
        score = indicators["score"]
        threshold = indicators["threshold"]
        ready = np.isfinite(score) & np.isfinite(threshold)
        candidate = np.sign(score.where(
            ready & score.abs().gt(threshold), 0.0
        )).astype(int).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        position = 0
        expiry = -1
        for i in range(len(data)):
            if position != 0:
                if i >= expiry:
                    position = 0
            elif candidate[i] != 0:
                position = int(candidate[i])
                expiry = i + params.holding_bars
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = WaveletParams()
    index = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(index))))
    data = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000000.0,
    }, index=index)
    full = GeneratedStrategy.indicators(None, data, params)
    prefix = GeneratedStrategy.indicators(None, data.iloc[:300], params)
    pd.testing.assert_frame_equal(full.iloc[:300], prefix)
    assert np.isfinite(full["score"].iloc[-1])
    manual = pd.DataFrame({
        "score": [1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        "threshold": 0.5,
    }, index=index[:8])
    result = GeneratedStrategy.generate_signals(
        None, data.iloc[:8], manual, None, params
    ).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert result["size"].gt(0).all()
