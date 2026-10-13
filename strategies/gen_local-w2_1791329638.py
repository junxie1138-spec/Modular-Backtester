from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityBalanceParams:
    vol_window: int = 20
    percentile_window: int = 126
    entry_percentile: float = 0.80
    profit_target: float = 0.045
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[VolatilityBalanceParams]):
    strategy_id = "gen_local-w2_1791329638"

    @classmethod
    def params_type(cls):
        return VolatilityBalanceParams

    def warmup_bars(self, params):
        return int(params.vol_window + params.percentile_window + 1)

    def indicators(self, data, params):
        if params.vol_window < 2 or params.percentile_window < 2:
            raise ValueError("Lookback windows must be at least two bars.")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between zero and one.")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["upside_vol"] = out["return"].clip(lower=0).rolling(params.vol_window).std(ddof=0)
        out["downside_vol"] = out["return"].clip(upper=0).rolling(params.vol_window).std(ddof=0)
        out["vol_sum"] = out["upside_vol"] + out["downside_vol"]
        out["balance"] = (out["upside_vol"] - out["downside_vol"]) / out["vol_sum"].replace(0, np.nan)
        out["threshold"] = out["balance"].rolling(params.percentile_window).quantile(params.entry_percentile).shift(1)
        out["drift"] = out["return"].rolling(params.vol_window).mean()
        out["above"] = out["balance"].gt(out["threshold"]) & out["drift"].gt(0)
        out["entry"] = out["above"] & ~out["above"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        if params.max_hold_bars < 1 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Holding period and profit target must be positive.")
        signal = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            # A decision on the previous close becomes a position at this open.
            if active and entry_bar == -1:
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    raise ValueError("Entry open must be finite and positive.")
                entry_bar = i
                entry_price = opens[i]
            if active:
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or expired:
                    active = False
                    entry_bar = -1
                    continue
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = -1
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    data = pd.DataFrame({"open": [200., 100., 101., 101., 101.], "close": [200., 100., 101., 101., 101.]},
                        index=pd.date_range("2020-01-01", periods=5))
    ind = pd.DataFrame({"entry": [True, False, False, False, False]}, index=data.index)
    params = VolatilityBalanceParams(profit_target=0.05, max_hold_bars=3)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    data.loc[data.index[2], "close"] = 106.
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
