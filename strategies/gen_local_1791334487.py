from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureRecoveryParams:
    excursion_window: int = 63
    area_window: int = 8
    minimum_excursion: float = 0.03
    area_threshold: float = 0.06
    profit_target: float = 0.04
    time_stop: int = 20


class GeneratedStrategy(BaseStrategy[SignatureRecoveryParams]):
    strategy_id = "gen_local_1791334487"

    @classmethod
    def params_type(cls) -> type[SignatureRecoveryParams]:
        return SignatureRecoveryParams

    def warmup_bars(self, params: SignatureRecoveryParams) -> int:
        return int(params.excursion_window + params.area_window + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureRecoveryParams) -> pd.DataFrame:
        if params.excursion_window < 2 or params.area_window < 2:
            raise ValueError("Indicator windows must be at least two bars")
        if params.minimum_excursion <= 0 or params.area_threshold < 0:
            raise ValueError("Excursion must be positive and area threshold nonnegative")
        if params.profit_target <= 0 or params.time_stop < 1:
            raise ValueError("Profit target and time stop must be positive")

        out = pd.DataFrame(index=data.index)
        out["peak"] = data["close"].rolling(params.excursion_window).max()
        out["trough"] = data["close"].rolling(params.excursion_window).min()
        out["drawdown"] = 1.0 - data["close"] / out["peak"].where(out["peak"] > 0)
        out["runup"] = data["close"] / out["trough"].where(out["trough"] > 0) - 1.0
        out["range"] = data["high"] - data["low"]
        out["pressure"] = (
            (2.0 * data["close"] - data["high"] - data["low"])
            / out["range"].where(out["range"] > 0)
        ).clip(-1.0, 1.0)
        out.loc[out["range"].eq(0), "pressure"] = 0.0
        out["valid_bar"] = (
            np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
            & data[["open", "high", "low", "close"]].gt(0).all(axis=1)
            & data["high"].ge(data["low"])
            & data["close"].between(data["low"], data["high"])
        )
        out["pressure"] = out["pressure"].where(out["valid_bar"])

        for side, depth, sign in (("long", "drawdown", 1.0), ("short", "runup", -1.0)):
            out[side + "_dx"] = out[depth].diff()
            out[side + "_dy"] = sign * out["pressure"]
            out[side + "_y"] = out[side + "_dy"].fillna(0.0).cumsum()
            out[side + "_cross"] = (
                out[depth].shift(1) * out[side + "_dy"]
                - out[side + "_y"].shift(1).fillna(0.0) * out[side + "_dx"]
            )
            out[side + "_sum_dx"] = out[side + "_dx"].rolling(params.area_window).sum()
            out[side + "_sum_dy"] = out[side + "_dy"].rolling(params.area_window).sum()
            out[side + "_scale"] = (
                out[side + "_dx"].abs().rolling(params.area_window).sum()
                * out[side + "_dy"].abs().rolling(params.area_window).sum()
            )
            out[side + "_area"] = 0.5 * (
                out[side + "_cross"].rolling(params.area_window).sum()
                - out[depth].shift(params.area_window) * out[side + "_sum_dy"]
                + out[side + "_y"].shift(params.area_window) * out[side + "_sum_dx"]
            ) / out[side + "_scale"].where(out[side + "_scale"] > 1e-12)
            out[side + "_ready"] = (
                out["valid_bar"].astype(int).rolling(params.excursion_window + params.area_window).sum()
                .eq(params.excursion_window + params.area_window)
                & np.isfinite(out[side + "_area"])
            )
            out[side + "_setup"] = (
                out[side + "_ready"]
                & out[depth].ge(params.minimum_excursion)
                & out[side + "_dx"].lt(0.0)
                & out[side + "_dy"].gt(0.0)
                & out[side + "_area"].ge(params.area_threshold)
            ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureRecoveryParams,
    ) -> SignalFrame:
        long_setup = indicators["long_setup"].fillna(False).astype(bool)
        short_setup = indicators["short_setup"].fillna(False).astype(bool)
        long_confirmed = (long_setup & long_setup.shift(1, fill_value=False)).to_numpy()
        short_confirmed = (short_setup & short_setup.shift(1, fill_value=False)).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                        continue
                gain = position * (closes[i] / entry_price - 1.0)
                expired = i - entry_bar + 1 >= params.time_stop
                if expired or (np.isfinite(gain) and gain >= params.profit_target):
                    position = 0
                    entry_price = np.nan
                    continue
            elif long_confirmed[i] != short_confirmed[i]:
                position = 1 if long_confirmed[i] else -1
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({
        "open": 100.0,
        "high": 104.0,
        "low": 96.0,
        "close": [100.0, 100.0, 100.0, 103.0, 100.0, 100.0, 100.0, 97.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    setups = pd.DataFrame({
        "long_setup": [True, True, False, False, False, False, False, False, False],
        "short_setup": [False, False, False, False, True, True, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    params = SignatureRecoveryParams(profit_target=0.02, time_stop=3)
    result = strategy.generate_signals(data, setups, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, -1, -1, 0]
    flat = data.assign(close=100.0)
    setups["short_setup"] = False
    result = strategy.generate_signals(flat, setups, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0]
    derived = strategy.indicators(flat, params)
    assert not derived[["long_setup", "short_setup"]].to_numpy().any()


if __name__ == "__main__":
    _self_check()
