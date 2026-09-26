from datetime import datetime, timezone

import pandas as pd

from src.market_data.base import MarketDataProvider, Quote
from src.signals.models import (
    MarketRegime,
    PriceFreshness,
    ScoreBreakdown,
    SignalAction,
    TechnicalSnapshot,
    TradeSignal,
)
from src.signals.service import SignalService


class FakeProvider(MarketDataProvider):
    name = "fake"

    def quote(self, symbol: str) -> Quote:
        return Quote(symbol=symbol, price=150.0, timestamp=datetime.now(timezone.utc), source=self.name)

    def history(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        n = 260
        base = [100 + i * 0.2 for i in range(n)]
        return pd.DataFrame(
            {
                "open": [x - 0.2 for x in base],
                "high": [x + 1 for x in base],
                "low": [x - 1 for x in base],
                "close": base,
                "volume": [1_000_000 + i * 1000 for i in range(n)],
            }
        )


def test_trade_signal_contract_accepts_detailed_levels():
    now = datetime.now(timezone.utc)
    signal = TradeSignal(
        symbol="AAPL",
        action=SignalAction.long,
        generated_at=now,
        price_timestamp=now,
        price_freshness=PriceFreshness.unknown,
        timeframe="swing",
        market_regime=MarketRegime.uptrend,
        current_price=200.0,
        entry_zone_low=197.5,
        entry_zone_high=200.0,
        ideal_entry=198.5,
        stop_loss=193.0,
        take_profit_1=204.0,
        take_profit_2=209.5,
        take_profit_3=215.0,
        risk_reward=2.0,
        signal_strength=76.0,
        scores=ScoreBreakdown(trend=20, momentum=15, volume=12, structure=12, regime=10, risk_reward=7),
        technicals=TechnicalSnapshot(rsi_14=58.0),
        thesis=["Trend aligned"],
        risks=["Stop can be hit"],
        invalidation="Daily close below 193.00",
        data_source="test",
    )
    assert signal.scores.total == 76
    assert signal.take_profit_3 == 215.0


def test_signal_service_builds_risk_defined_setup():
    signal = SignalService(FakeProvider()).generate("TEST", portfolio_value=10_000, max_risk_pct=1.0)
    assert signal.symbol == "TEST"
    assert signal.data_source == "fake"
    assert signal.market_regime in set(MarketRegime)
    assert len(signal.timeframes) >= 1
    if signal.action in {SignalAction.long, SignalAction.strong_long}:
        assert signal.stop_loss < signal.ideal_entry < signal.take_profit_2
        assert signal.position_plan.max_risk_amount == 100.0
        assert signal.position_plan.estimated_position_value <= 10_000


def test_non_executable_signal_has_no_fake_levels():
    service = SignalService(FakeProvider())
    levels = service._levels(
        SignalAction.neutral,
        150.0,
        {"atr_14": 2.0, "support": 140.0, "resistance": 160.0, "ema_20": 149.0},
    )
    assert levels["ideal_entry"] is None
    assert levels["stop_loss"] is None
    assert levels["take_profit_1"] is None
