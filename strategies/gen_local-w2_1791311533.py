from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletTransferParams:
    percentile_window: int = 126
    entry_quantile: float = 0.75
    vol_window: int = 20
    target_vol: float = 0.10


class GeneratedStrategy(BaseStrategy[WaveletTransferParams]):
    strategy_id = "gen_local-w2_1791311533"

    @classmethod
    def params_type(cls) -> type[WaveletTransferParams]:
        return WaveletTransferParams

    def warmup_bars(self, params: WaveletTransferParams) -> int:
        return int(max(params.percentile_window + 10, params.vol_window + 1))

    def indicators(self, data: pd.DataFrame, params: WaveletTransferParams) -> pd.DataFrame:
        if params.percentile_window < 2 or params.vol_window < 2:
            raise ValueError("Lookback windows must be at least two.")
        if not 0.5 < params.entry_quantile < 1.0 or not np.isfinite(params.target_vol) or params.target_vol <= 0:
            raise ValueError("Invalid percentile or volatility target.")
        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        out["body"] = np.log(closing / opening).replace([np.inf, -np.inf], np.nan)
        out["gap"] = np.log(opening / closing.shift(1)).replace([np.inf, -np.inf], np.nan)
        out["return"] = np.log(closing / closing.shift(1)).replace([np.inf, -np.inf], np.nan)
        for component in ("body", "gap"):
            energy = pd.Series(0.0, index=data.index)
            for half in (1, 2, 4):
                block = out[component].rolling(half, min_periods=half).sum()
                detail = (block - block.shift(half)) / np.sqrt(2.0 * half)
                out[f"{component}_haar_{half}"] = detail
                energy = energy + detail.pow(2)
            out[f"{component}_energy"] = energy
        out["total_energy"] = out["body_energy"] + out["gap_energy"]
        out["intraday_share"] = out["body_energy"] / out["total_energy"].where(out["total_energy"] > 0)
        out["transfer"] = out["intraday_share"].diff()
        out["transfer_threshold"] = out["transfer"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.entry_quantile)
        out["annual_vol"] = out["return"].rolling(
            params.vol_window, min_periods=params.vol_window
        ).std(ddof=0) * np.sqrt(252.0)
        out["exposure"] = (params.target_vol / out["annual_vol"].where(
            out["annual_vol"] > 0
        )).clip(lower=0.01, upper=0.95).fillna(0.01)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletTransferParams,
    ) -> SignalFrame:
        ready = indicators[["transfer", "transfer_threshold", "body", "annual_vol"]].notna().all(axis=1)
        # A positive energy handoff must exceed its own historical percentile.
        entry = (
            ready
            & (indicators["transfer"] > indicators["transfer_threshold"])
            & (indicators["transfer"] > 0)
            & (indicators["body"] > 0)
            & (indicators["annual_vol"] > 0)
        )
        df = pd.DataFrame(index=data.index)
        # Retain exposure exactly while entry is true; its reversal is the only exit.
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["exposure"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = WaveletTransferParams()
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    rng = np.random.default_rng(7)
    n = 400
    body = rng.normal(0.001, 0.009, n)
    gap = rng.normal(0.0, 0.004, n)
    close = 100.0 * np.exp(np.cumsum(body + gap))
    opening = close * np.exp(-body)
    data = pd.DataFrame({
        "open": opening, "high": np.maximum(opening, close) * 1.001,
        "low": np.minimum(opening, close) * 0.999,
        "close": close, "volume": np.full(n, 1000000.0),
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    ind = strategy.indicators(data, p)
    signals = strategy.generate_signals(data, ind, None, p).data
    expected = ((ind["transfer"] > ind["transfer_threshold"]) & (ind["transfer"] > 0)
                & (ind["body"] > 0) & (ind["annual_vol"] > 0)).astype(int)
    assert signals["signal"].equals(expected.shift(1).fillna(0).astype(int))
    assert signals["signal"].sum() > 0 and signals["signal"].isin([0, 1]).all()
    assert np.isfinite(signals["size"]).all() and signals["size"].between(0.01, 0.95).all()
    assert not signals["signal"].iloc[:strategy.warmup_bars(p)].any()
    prefix = data.iloc[:300]
    prefix_signals = strategy.generate_signals(prefix, strategy.indicators(prefix, p), None, p).data
    pd.testing.assert_frame_equal(signals.iloc[:300], prefix_signals)
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    assert not strategy.generate_signals(flat, strategy.indicators(flat, p), None, p).data["signal"].any()
