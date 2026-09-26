from datetime import datetime, timezone

from src.signals.models import (
    ScoreBreakdown,
    SignalAction,
    TechnicalSnapshot,
    TradeSignal,
)


def test_trade_signal_contract_accepts_detailed_levels():
    signal = TradeSignal(
        symbol="AAPL",
        action=SignalAction.long,
        generated_at=datetime.now(timezone.utc),
        timeframe="swing",
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
        scores=ScoreBreakdown(
            trend=20,
            momentum=15,
            volume=12,
            structure=12,
            regime=10,
            risk_reward=7,
        ),
        technicals=TechnicalSnapshot(rsi_14=58.0),
        thesis=["Trend aligned"],
        risks=["Stop can be hit"],
        invalidation="Daily close below 193.00",
        data_source="test",
    )

    assert signal.symbol == "AAPL"
    assert signal.scores.total == 76
    assert signal.take_profit_3 == 215.0
