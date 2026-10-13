from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureRecoveryParams:
    lookback: int = 12
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[SignatureRecoveryParams]):
    strategy_id = "gen_local_1791343109"

    @classmethod
    def params_type(cls) -> type[SignatureRecoveryParams]:
        return SignatureRecoveryParams

    @staticmethod
    def warmup_bars(params: SignatureRecoveryParams) -> int:
        if params.lookback < 2 or params.holding_bars < 1:
            raise ValueError("lookback must be >= 2 and holding_bars >= 1")
        return int(63 + params.lookback + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureRecoveryParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        length = params.lookback
        out = pd.DataFrame(index=data.index)
        out["close"] = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out["close"] = out["close"].where(out["close"] > 0)
        out["volume"] = data["volume"].astype(float).replace([np.inf, -np.inf], np.nan).clip(lower=0)
        out["peak"] = out["close"].rolling(63, min_periods=63).max()
        out["depth"] = 1.0 - out["close"] / out["peak"]
        out["price_change"] = out["close"].diff()
        out["volume_mean"] = out["volume"].rolling(20, min_periods=20).mean()
        out["selling_increment"] = (
            out["volume"] / out["volume_mean"].replace(0.0, np.nan)
        ).where(out["price_change"] < 0, 0.0).fillna(0.0)
        out["selling_clock"] = out["selling_increment"].cumsum()
        out["depth_increment"] = out["depth"].diff()
        out["cross_increment"] = (
            out["depth"].shift(1) * out["selling_increment"]
            - out["selling_clock"].shift(1) * out["depth_increment"]
        )
        out["selling_total"] = out["selling_increment"].rolling(length, min_periods=length).sum()
        out["depth_total"] = out["depth"] - out["depth"].shift(length)
        out["depth_variation"] = out["depth_increment"].abs().rolling(length, min_periods=length).sum()
        # Translation-invariant antisymmetric second-level signature area.
        out["area"] = 0.5 * (
            out["cross_increment"].rolling(length, min_periods=length).sum()
            - out["depth"].shift(length) * out["selling_total"]
            + out["selling_clock"].shift(length) * out["depth_total"]
        )
        out["area_scale"] = out["selling_total"] * out["depth_variation"]
        out["area_score"] = (
            out["area"] / out["area_scale"].where(out["area_scale"] > 1e-12)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["recovering"] = (out["depth"] < out["depth"].shift(3)).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureRecoveryParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        score = indicators["area_score"].fillna(0.0)
        entries = (
            (score > 0.10)
            & (score.shift(1).fillna(0.0) <= 0.10)
            & indicators["depth"].between(0.02, 0.15)
            & indicators["recovering"].fillna(False)
            & (indicators["price_change"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for bar in range(len(data)):
            if exit_bar >= 0:
                if bar < exit_bar:
                    signal[bar] = 1
                else:
                    # Always emit zero on expiry, even if an entry also fires.
                    exit_bar = -1
            elif entries[bar]:
                signal[bar] = 1
                exit_bar = bar + params.holding_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = SignatureRecoveryParams()
    count = params.holding_bars + 5
    index = pd.date_range("2020-01-01", periods=count, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"area_score": 0.2, "depth": 0.05, "recovering": True, "price_change": 1.0},
        index=index,
    )
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    expected = np.zeros(count, dtype=int)
    expected[1:params.holding_bars + 1] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert (result["size"] > 0).all()
    flat_indicators = GeneratedStrategy.indicators(data, params)
    flat_result = GeneratedStrategy.generate_signals(data, flat_indicators, None, params).data
    assert flat_result["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
