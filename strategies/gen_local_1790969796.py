from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureCompressionParams:
    path_window: int = 8
    compression_window: int = 5
    reference_window: int = 40
    compression_limit: float = 0.85
    area_threshold: float = 0.08


class GeneratedStrategy(BaseStrategy[SignatureCompressionParams]):
    strategy_id = "gen_local_1790969796"

    @classmethod
    def params_type(cls):
        return SignatureCompressionParams

    def warmup_bars(self, params: SignatureCompressionParams) -> int:
        return int(max(200, params.path_window + 1, params.compression_window, params.reference_window))

    def indicators(self, data: pd.DataFrame, params: SignatureCompressionParams) -> pd.DataFrame:
        if params.path_window < 2 or params.compression_window < 1:
            raise ValueError("Path window must be at least two and compression window positive.")
        if params.reference_window < params.compression_window:
            raise ValueError("Reference window must cover the compression window.")
        if not 0.0 < params.compression_limit <= 1.0:
            raise ValueError("Compression limit must be in (0, 1].")
        if not 0.0 <= params.area_threshold < 0.5:
            raise ValueError("Area threshold must be in [0, 0.5).")

        out = pd.DataFrame(index=data.index)
        out["safe_close"] = data["close"].where(data["close"] > 0).replace([np.inf, -np.inf], np.nan)
        out["path_price"] = np.log(out["safe_close"])
        out["path_range"] = ((data["high"] - data["low"]) / out["safe_close"]).where(data["high"] >= data["low"]).replace([np.inf, -np.inf], np.nan)
        out["price_step"] = out["path_price"].diff()
        out["range_step"] = out["path_range"].diff()
        out["cross_step"] = out["path_price"].shift(1) * out["range_step"] - out["path_range"].shift(1) * out["price_step"]
        w = params.path_window
        out["cross_sum"] = out["cross_step"].rolling(w, min_periods=w).sum()
        out["price_start"] = out["path_price"].shift(w)
        out["range_start"] = out["path_range"].shift(w)
        out["net_price"] = out["path_price"] - out["price_start"]
        out["net_range"] = out["path_range"] - out["range_start"]
        # Antisymmetric second-order signature of the translated price-range path.
        out["signed_area"] = 0.5 * (out["cross_sum"] - out["price_start"] * out["net_range"] + out["range_start"] * out["net_price"])
        out["price_variation"] = out["price_step"].abs().rolling(w, min_periods=w).sum()
        out["range_variation"] = out["range_step"].abs().rolling(w, min_periods=w).sum()
        out["area_scale"] = out["price_variation"] * out["range_variation"]
        out["signature_area"] = out["signed_area"] / out["area_scale"].where(out["area_scale"] > 0)
        out["recent_range"] = out["path_range"].rolling(params.compression_window, min_periods=params.compression_window).mean()
        out["reference_range"] = out["path_range"].rolling(params.reference_window, min_periods=params.reference_window).mean()
        out["compression"] = out["recent_range"] / out["reference_range"].where(out["reference_range"] > 0)
        out["ma200"] = out["safe_close"].rolling(200, min_periods=200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureCompressionParams) -> SignalFrame:
        entry_condition = (
            (indicators["signature_area"] > params.area_threshold)
            & (indicators["net_price"] > 0)
            & (indicators["compression"] < params.compression_limit)
            & (data["close"] > indicators["ma200"])
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Remain long only while the full entry condition holds; its reversal exits.
        df["signal"] = entry_condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = SignatureCompressionParams()
    strategy = GeneratedStrategy()
    t = np.arange(240, dtype=float)
    close = np.exp(4.5 + 0.0005 * t + 0.005 * np.sin(t / 3.0))
    width = 0.006 + 0.002 * np.sin(t / 5.0)
    data = pd.DataFrame({"open": close, "high": close * (1 + width / 2), "low": close * (1 - width / 2), "close": close, "volume": np.full(len(t), 1000000.0)}, index=pd.date_range("2020-01-01", periods=len(t), freq="B"))
    ind = strategy.indicators(data, params)
    w = params.path_window
    x = ind["path_price"].iloc[-w-1:].to_numpy()
    y = ind["path_range"].iloc[-w-1:].to_numpy()
    x = x - x[0]
    y = y - y[0]
    expected = 0.5 * np.sum(x[:-1] * np.diff(y) - y[:-1] * np.diff(x))
    expected /= np.abs(np.diff(x)).sum() * np.abs(np.diff(y)).sum()
    assert np.isclose(ind["signature_area"].iloc[-1], expected, atol=1e-10)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    signals = strategy.generate_signals(data, ind, ctx, params).data
    assert (signals["signal"].iloc[:200] == 0).all()
    probe = ind.iloc[-5:].copy()
    probe["signature_area"] = [0.0, 0.2, 0.2, 0.0, 0.2]
    probe["net_price"] = 0.01
    probe["compression"] = 0.5
    probe["ma200"] = data["close"].iloc[-5:] * 0.9
    result = strategy.generate_signals(data.iloc[-5:], probe, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
