from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    lookback: int = 8
    volume_window: int = 20
    area_threshold: float = 0.08
    profit_target: float = 0.012
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791264023"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(params.volume_window + params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if (
            params.lookback < 2 or params.volume_window < 2
            or not 0.0 < params.area_threshold < 0.5
            or not 0.0 < params.profit_target < 1.0
            or not 3 <= params.hold_bars <= 5
        ):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        n = params.lookback
        out["return"] = np.log(data["close"].where(data["close"] > 0)).diff()
        out["volume_mean"] = data["volume"].rolling(params.volume_window).mean()
        out["activity"] = data["volume"] / out["volume_mean"].replace(0.0, np.nan) - 1.0
        out["price_path"] = out["return"].fillna(0.0).cumsum()
        out["activity_path"] = out["activity"].fillna(0.0).cumsum()
        out["cross_increment"] = (
            out["price_path"].shift(1).fillna(0.0) * out["activity"]
            - out["activity_path"].shift(1).fillna(0.0) * out["return"]
        )
        out["momentum"] = out["return"].rolling(n).sum()
        out["net_activity"] = out["activity"].rolling(n).sum()
        out["price_variation"] = out["return"].abs().rolling(n).sum()
        out["activity_variation"] = out["activity"].abs().rolling(n).sum()
        # Translation-invariant second-order antisymmetric signature.
        out["area"] = 0.5 * (
            out["cross_increment"].rolling(n).sum()
            - out["price_path"].shift(n).fillna(0.0) * out["net_activity"]
            + out["activity_path"].shift(n).fillna(0.0) * out["momentum"]
        )
        out["normalized_area"] = out["area"] / (
            out["price_variation"] * out["activity_variation"]
        ).replace(0.0, np.nan)
        # Positive leadership means excess volume precedes directional price movement.
        out["leadership"] = -out["normalized_area"] * np.sign(out["momentum"])
        out["candidate"] = np.where(
            (out["leadership"] >= params.area_threshold) & (out["net_activity"] > 0.0),
            np.sign(out["momentum"]).fillna(0.0),
            0,
        ).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        candidate = indicators["candidate"].fillna(0).astype(int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry else np.nan
                if (
                    not valid_entry
                    or gain >= params.profit_target
                    or i - entry_bar + 1 >= params.hold_bars
                ):
                    position = 0
            elif confirmed[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(confirmed[i])
                entry_bar = i + 1
                entry_price = np.nan
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 97.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction in (1, -1):
        ind = pd.DataFrame({"candidate": [direction, direction, 0, 0, 0, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, 0, direction, direction, direction, direction, 0, 0, 0]
        target_data = data.copy()
        target_data.loc[index[3], "close"] = 100.0 * (1.0 + direction * 0.02)
        result = strategy.generate_signals(target_data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, 0, direction, direction, 0, 0, 0, 0, 0]
    ind = pd.DataFrame({"candidate": [1, 0, -1, 0, 1, 0, -1, 0, 0]}, index=index)
    assert not strategy.generate_signals(data, ind, ctx, params).data["signal"].any()
    assert not strategy.indicators(data, params)["candidate"].any()
