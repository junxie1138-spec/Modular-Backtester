from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionShiftParams:
    baseline_bars: int = 20
    auction_bars: int = 3
    volume_ratio: float = 1.4
    shock_sigma: float = 0.8
    prior_location_ceiling: float = 0.45
    acceptance_floor: float = 0.8
    profit_target: float = 0.012
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[AuctionShiftParams]):
    strategy_id = "gen_local-w2_1791321976"

    @classmethod
    def params_type(cls) -> type[AuctionShiftParams]:
        return AuctionShiftParams

    def warmup_bars(self, params: AuctionShiftParams) -> int:
        return int(max(params.baseline_bars + 2, params.auction_bars + 1))

    def indicators(self, data: pd.DataFrame, params: AuctionShiftParams) -> pd.DataFrame:
        if params.baseline_bars < 2 or params.auction_bars < 1:
            raise ValueError("Lookbacks must be positive; baseline_bars must be at least two")
        if params.volume_ratio <= 0 or params.shock_sigma < 0:
            raise ValueError("Invalid volume or shock threshold")
        if not 0 <= params.prior_location_ceiling < params.acceptance_floor <= 1:
            raise ValueError("Invalid auction location thresholds")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] > 0)
        out["return"] = close.div(close.shift(1)).sub(1).replace([np.inf, -np.inf], np.nan)
        out["volume_mean"] = volume.rolling(params.baseline_bars).mean().shift(1)
        out["relative_volume"] = volume.div(out["volume_mean"])
        out["weighted_return"] = out["return"].mul(volume)
        out["prior_weighted_return"] = (
            out["weighted_return"].rolling(params.baseline_bars).sum().shift(1)
            .div(volume.rolling(params.baseline_bars).sum().shift(1))
        )
        out["prior_return_std"] = out["return"].rolling(params.baseline_bars).std(ddof=0).shift(1)
        out["return_innovation"] = out["return"] - out["prior_weighted_return"]
        out["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["close_location"] = (data["close"] - data["low"]).div(out["range"])
        out["prior_close_location"] = out["close_location"].rolling(params.auction_bars).mean().shift(1)
        out["volume_changepoint"] = (
            out["prior_weighted_return"].lt(0)
            & out["return"].gt(0)
            & out["return_innovation"].gt(params.shock_sigma * out["prior_return_std"])
            & out["relative_volume"].ge(params.volume_ratio)
        ).fillna(False)
        out["auction_shift"] = (
            out["prior_close_location"].le(params.prior_location_ceiling)
            & out["close_location"].ge(params.acceptance_floor)
            & out["close_location"].le(1)
            & data["close"].gt(data["open"])
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AuctionShiftParams,
    ) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be one or two")
        entries = (
            indicators["volume_changepoint"].fillna(False).astype(bool)
            & indicators["auction_shift"].fillna(False).astype(bool)
        ).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        held_bars = 0
        for i in range(len(data)):
            if active:
                # The preceding close requested entry; this session's open anchors the target.
                if held_bars == 0:
                    entry_price = opens[i]
                held_bars += 1
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i]) and closes[i] > 0
                target_hit = valid and closes[i] >= entry_price * (1 + params.profit_target)
                if not valid or target_hit or held_bars >= params.max_hold_bars:
                    active = False
                    held_bars = 0
                    continue
                decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                held_bars = 0
                entry_price = np.nan
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=4)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"volume_changepoint": [True, False, False, False], "auction_shift": [True, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    params = AuctionShiftParams(profit_target=0.01)
    result = strategy.generate_signals(data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0]
    data.loc[index[1], "close"] = 101.5
    result = strategy.generate_signals(data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 0, 0]
    indicators["auction_shift"] = False
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
