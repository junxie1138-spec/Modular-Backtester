from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    vol_window: int = 20
    shock_window: int = 5
    turnover_window: int = 20
    impact_window: int = 3
    atr_window: int = 14
    stress_threshold: float = 1.2
    impact_decay: float = 0.85
    breakeven_pct: float = 0.015
    trail_atr: float = 2.0
    initial_stop_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791255838"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.vol_window + 2,
                       params.shock_window + 2,
                       params.turnover_window + params.impact_window,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["downside_square"] = out["return"].clip(upper=0).pow(2)
        out["volatility"] = out["return"].rolling(params.vol_window).std(ddof=0)
        out["downside_rms"] = np.sqrt(out["downside_square"].rolling(params.shock_window).mean())
        out["stress"] = out["downside_rms"] / out["volatility"].replace(0, np.nan)
        out["dollar_volume"] = data["close"] * data["volume"]
        out["normal_turnover"] = out["dollar_volume"].rolling(params.turnover_window).mean()
        out["relative_turnover"] = out["dollar_volume"] / out["normal_turnover"].replace(0, np.nan)
        out["downside_impact"] = -out["return"].clip(upper=0) / out["relative_turnover"].where(out["relative_turnover"] > 0)
        out["impact"] = out["downside_impact"].rolling(params.impact_window).mean()
        out["recent_return"] = data["close"].pct_change(params.shock_window, fill_method=None)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["volatility_agrees"] = ((out["stress"] >= params.stress_threshold) &
                                    (out["stress"] < out["stress"].shift(1))).fillna(False)
        out["impact_agrees"] = ((out["impact"].shift(1) > 0) &
                                (out["impact"] < params.impact_decay * out["impact"].shift(1)) &
                                (out["recent_return"] < 0)).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        entries = (indicators["volatility_agrees"].fillna(False) &
                   indicators["impact_agrees"].fillna(False)).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        active = pending = armed = False
        entry = stop = peak = 0.0
        age = 0
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if pending:
                # The previous close's order fills at this bar's open.
                entry = opens[i]
                stop = entry - params.initial_stop_atr * atrs[i - 1]
                peak = entry
                active, pending, armed = True, False, False
                age = 0
            if active:
                age += 1
                valid = np.isfinite([opens[i], highs[i], lows[i], closes[i], atrs[i]]).all() and atrs[i] > 0
                exit_now = not valid or lows[i] <= stop or age >= params.max_hold
                if not exit_now:
                    peak = max(peak, closes[i])
                    if highs[i] >= entry * (1.0 + params.breakeven_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atrs[i])
                        exit_now = closes[i] <= stop
                if exit_now:
                    active = False
                else:
                    raw[i] = 1
            elif (i >= warmup and entries[i] and
                  np.isfinite([opens[i], highs[i], lows[i], closes[i], atrs[i]]).all() and
                  closes[i] > 0 and opens[i] > 0 and atrs[i] > 0):
                raw[i] = 1
                pending = True
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        # Stop decisions also execute next open; no intrabar fills are assumed.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=32, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5,
                         "close": 100.0, "volume": 1000000.0}, index=index)
    features = strategy.indicators(data, params)
    features["volatility_agrees"] = False
    features["impact_agrees"] = False
    features["atr"] = 1.0
    features.loc[index[23], ["volatility_agrees", "impact_agrees"]] = True
    # Arm at +2%; then widen ATR. The breakeven stop must not move down.
    data.loc[index[24], ["high", "low", "close"]] = [102.5, 99.5, 102.0]
    data.loc[index[25], ["high", "low", "close"]] = [102.5, 100.5, 102.0]
    features.loc[index[25], "atr"] = 5.0
    data.loc[index[26], ["high", "low", "close"]] = [101.5, 99.9, 101.0]
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].iloc[23:28].tolist() == [0, 1, 1, 1, 0]
    features["impact_agrees"] = False
    assert strategy.generate_signals(data, features, ctx, params).data["signal"].eq(0).all()
    assert result["size"].gt(0).all() and result["signal"].isin([0, 1]).all()
