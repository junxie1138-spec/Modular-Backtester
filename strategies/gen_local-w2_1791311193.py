from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    percentile_window: int = 126
    interference_quantile: float = 0.80
    volatility_window: int = 20
    target_daily_volatility: float = 0.006
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local-w2_1791311193"

    @classmethod
    def params_type(cls) -> type[WaveParams]:
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        return int(max(params.percentile_window + 10, params.volatility_window + 1))

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        if params.percentile_window < 2 or params.volatility_window < 2:
            raise ValueError("Rolling windows must be at least two bars")
        if not 0.0 < params.interference_quantile < 1.0:
            raise ValueError("Interference quantile must lie between zero and one")
        if params.target_daily_volatility <= 0.0 or params.profit_target <= 0.0:
            raise ValueError("Volatility target and profit target must be positive")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("Holding horizon must be one or two bars")

        out = pd.DataFrame(index=data.index)
        out["log_return"] = np.log(data["close"].where(data["close"] > 0.0)).diff()
        for scale in (1, 2, 4):
            mean = out["log_return"].rolling(scale, min_periods=scale).mean()
            out[f"haar_{scale}"] = np.sqrt(scale / 2.0) * (mean - mean.shift(scale))
        waves = out[["haar_1", "haar_2", "haar_4"]]
        out["energy"] = waves.pow(2).sum(axis=1, min_count=3)
        out["signed_sum"] = waves.sum(axis=1, min_count=3)
        out["interference"] = (
            1.0 - out["signed_sum"].pow(2) / (3.0 * out["energy"].where(out["energy"] > 0.0))
        ).clip(0.0, 1.0)
        out["interference_threshold"] = out["interference"].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.interference_quantile).shift(1)
        out["positive_phase"] = waves.gt(0.0).all(axis=1)
        out["entry"] = (
            out["interference"].shift(1).gt(out["interference_threshold"].shift(1))
            & out["positive_phase"]
            & ~out["positive_phase"].shift(1, fill_value=False)
        ).fillna(False)
        out["realized_volatility"] = out["log_return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0)
        out["wavelet_volatility"] = np.sqrt(out["energy"] / 3.0)
        out["risk_volatility"] = np.maximum(
            out["realized_volatility"], out["wavelet_volatility"]
        )
        out["target_size"] = (
            params.target_daily_volatility / out["risk_volatility"].where(out["risk_volatility"] > 0.0)
        ).replace([np.inf, -np.inf], np.nan).clip(0.01, 0.95).fillna(0.01)
        out["entry"] &= out["risk_volatility"].gt(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.01, dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        proposed_sizes = indicators["target_size"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        age = 0
        entry_price = 0.0
        held_size = 0.01

        for i in range(len(data)):
            if active:
                # The previous close's entry decision fills at this bar's open.
                if age == 0:
                    entry_price = opens[i]
                age += 1
                sizes[i] = held_size
                valid = (
                    np.isfinite(entry_price) and entry_price > 0.0
                    and np.isfinite(closes[i]) and closes[i] > 0.0
                )
                target_hit = valid and closes[i] >= entry_price * (1.0 + params.profit_target)
                if not valid or target_hit or age >= params.max_hold_bars:
                    active = False
                    age = 0
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                held_size = float(proposed_sizes[i])
                if not np.isfinite(held_size) or held_size <= 0.0:
                    continue
                active = True
                age = 0
                signals[i] = 1
                sizes[i] = held_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0] * 6,
        "high": [102.0] * 6,
        "low": [99.0] * 6,
        "close": [100.0, 100.0, 100.0, 100.0, 101.0, 100.0],
        "volume": [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [True, False, False, True, False, False],
        "target_size": [0.4] * 6,
    }, index=index)
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, WaveParams()
    ).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 0]
    assert result["size"].gt(0.0).all()
    assert result.loc[result["signal"].eq(1), "size"].eq(0.4).all()


if __name__ == "__main__":
    _self_check()
