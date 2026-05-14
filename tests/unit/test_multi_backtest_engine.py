import pandas as pd


def _ohlcv(closes, start="2024-01-02"):
    idx = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "open": closes, "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes], "close": closes,
            "volume": [1_000_000] * len(closes),
        },
        index=idx,
    )


class _SimpleMRStrategy:
    """Minimal mean-reversion stub: signal=1 if close < mean10, else 0."""
    strategy_id = "test_mr_stub"
    uses_multi_symbol = True
    uses_per_bar = False

    def indicators(self, data, params):
        return pd.DataFrame({"mean10": data["close"].rolling(10).mean()}, index=data.index)

    def generate_signals_for_symbol(self, *, data, indicators, params):
        sig = (data["close"] < indicators["mean10"]).astype(float).shift(1).fillna(0)
        return pd.DataFrame({"signal": sig, "size": 1.0}, index=data.index)


def test_engine_runs_multi_symbol_end_to_end():
    from backtester.engine.multi_backtest_engine import MultiSymbolBacktestEngine
    from backtester.config.models import PortfolioConfig, ExecutionConfig
    from backtester.engine.broker import Broker
    from backtester.engine.multi_portfolio import MultiSymbolPortfolioSimulator

    closes_a = [100 - 0.5 * i if i < 15 else 100 - 7.5 + 0.5 * (i - 15) for i in range(30)]
    closes_b = [200 + 0.5 * i for i in range(30)]
    data = {"AAA": _ohlcv(closes_a), "BBB": _ohlcv(closes_b)}
    sectors = {"AAA": "X", "BBB": "Y"}

    sim = MultiSymbolPortfolioSimulator(
        config=PortfolioConfig(sizing_mode="percent_equity", size=0.1,
                               position_cap_pct=1.0, cash_reserve_pct=0.0,
                               risk_budget_pct=1.0, sector_cap_pct=1.0),
        initial_cash=100_000.0,
        broker_factory=lambda: Broker(ExecutionConfig(
            initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0,
        )),
    )
    engine = MultiSymbolBacktestEngine(simulator=sim)
    result = engine.run(
        strategy=_SimpleMRStrategy(),
        symbols=["AAA", "BBB"],
        data=data, sectors=sectors, aux_data={}, params=None, regime_config=None,
    )
    assert result.equity_curve is not None
    assert "AAA" in result.trades_per_symbol
    assert "BBB" in result.trades_per_symbol
