from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    signature_window: int = 12
    area_threshold: float = 0.02
    atr_window: int = 14
    vol_window: int = 20
    target_vol: float = 0.12
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1790978078"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(200, params.signature_window + 2,
                       params.atr_window + 1, params.vol_window + 1))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if (min(params.signature_window, params.atr_window, params.vol_window) < 2
                or params.hold_bars < 1
                or not 0.0 < params.area_threshold < 1.0
                or not np.isfinite(params.target_vol) or params.target_vol <= 0.0
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["ma200"] = close.rolling(200).mean()
        out["price_log"] = np.log(close)
        out["range_fraction"] = out["true_range"] / close
        out["range_log"] = np.log(out["range_fraction"].where(out["range_fraction"] > 0.0))
        out["return"] = out["price_log"].diff()
        out["range_change"] = out["range_log"].diff()
        x, y = out["price_log"], out["range_log"]
        length = params.signature_window
        out["edge_area"] = x.shift(1) * y - y.shift(1) * x
        # Close the rolling path with its endpoint-to-origin chord.
        out["signed_area"] = 0.5 * (
            out["edge_area"].rolling(length).sum()
            - x.shift(length) * y + y.shift(length) * x
        )
        out["price_variation"] = out["return"].abs().rolling(length).sum()
        out["range_variation"] = out["range_change"].abs().rolling(length).sum()
        out["area_scale"] = out["price_variation"] * out["range_variation"]
        out["area"] = out["signed_area"] / out["area_scale"].where(out["area_scale"] > 1e-12)
        out["displacement"] = x - x.shift(length)
        out["realized_vol"] = out["return"].rolling(params.vol_window).std(ddof=0) * np.sqrt(252.0)
        out["target_size"] = (
            params.target_vol / out["realized_vol"].where(out["realized_vol"] > 1e-12)
        ).clip(0.01, 1.0).fillna(0.01)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SignatureParams,
    ) -> SignalFrame:
        required = ["area", "displacement", "return", "ma200", "atr", "realized_vol", "target_size"]
        valid = np.isfinite(indicators[required].to_numpy()).all(axis=1)
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        area = indicators["area"].to_numpy()
        displacement = indicators["displacement"].to_numpy()
        returns = indicators["return"].to_numpy()
        ma = indicators["ma200"].to_numpy()
        atr = indicators["atr"].to_numpy()
        vol = indicators["realized_vol"].to_numpy()
        target_size = indicators["target_size"].to_numpy()
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.01, dtype=float)
        side, entry_bar = 0, -1
        pending = False
        stop, entry_atr, active_size = 0.0, 0.0, 0.01
        for i in range(len(data)):
            if side:
                if pending:
                    # Observe the entry open only when that bar arrives.
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        side, pending = 0, False
                        continue
                    stop = opens[i] - side * params.stop_atr * entry_atr
                    entry_bar, pending = i, False
                if (not np.isfinite(close[i]) or close[i] <= 0.0
                        or side * (close[i] - stop) < 0.0
                        or i - entry_bar + 1 >= params.hold_bars):
                    side = 0
                    continue
            elif valid[i] and np.isfinite(close[i]) and close[i] > 0.0 and atr[i] > 0.0 and vol[i] > 0.0:
                if (close[i] > ma[i] and displacement[i] < 0.0
                        and returns[i] > 0.0 and area[i] > params.area_threshold):
                    side = 1
                elif (close[i] < ma[i] and displacement[i] > 0.0
                        and returns[i] < 0.0 and area[i] < -params.area_threshold):
                    side = -1
                if side:
                    entry_atr, active_size = atr[i], target_size[i]
                    pending = True
            signals[i], sizes[i] = side, active_size if side else 0.01
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), SignatureParams()
    index = pd.date_range("2020-01-01", periods=8)
    close = np.array([100.0, 105.0, 99.0, 97.0, 100.0, 101.0, 103.0, 100.0])
    data = pd.DataFrame({"open": 100.0, "high": close + 1.0,
                         "low": close - 1.0, "close": close, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"area": [0.3, 0, 0, 0, -0.3, 0, 0, 0],
                        "displacement": [-0.01, 0, 0, 0, 0.01, 0, 0, 0],
                        "return": [0.01, 0, 0, 0, -0.01, 0, 0, 0],
                        "ma200": [90.0] * 4 + [110.0] * 4,
                        "atr": 1.0, "realized_vol": 0.2, "target_size": 0.6}, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]
    assert np.isfinite(result["size"]).all() and (result["size"] > 0.0).all()
    params.hold_bars = 2
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    assert strategy.warmup_bars(SignatureParams(signature_window=250)) == 252


if __name__ == "__main__":
    _self_check()
