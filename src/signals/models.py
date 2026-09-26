from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class SignalAction(str, Enum):
    strong_long = "STRONG_LONG"
    long = "LONG"
    watch = "WATCH"
    neutral = "NEUTRAL"
    short = "SHORT"
    strong_short = "STRONG_SHORT"
    no_trade = "NO_TRADE"


class SignalLifecycle(str, Enum):
    waiting = "WAITING"
    triggered = "TRIGGERED"
    tp1_hit = "TP1_HIT"
    tp2_hit = "TP2_HIT"
    tp3_hit = "TP3_HIT"
    stopped = "STOPPED"
    invalidated = "INVALIDATED"
    closed = "CLOSED"


class ScoreBreakdown(BaseModel):
    trend: float = Field(ge=0, le=25)
    momentum: float = Field(ge=0, le=20)
    volume: float = Field(ge=0, le=15)
    structure: float = Field(ge=0, le=15)
    regime: float = Field(ge=0, le=15)
    risk_reward: float = Field(ge=0, le=10)

    @property
    def total(self) -> float:
        return round(self.trend + self.momentum + self.volume + self.structure + self.regime + self.risk_reward, 2)


class TechnicalSnapshot(BaseModel):
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    ema_20: float | None = None
    ema_50: float | None = None
    ema_200: float | None = None
    atr_14: float | None = None
    relative_volume: float | None = None
    support: float | None = None
    resistance: float | None = None


class PositionPlan(BaseModel):
    portfolio_value: float | None = Field(default=None, gt=0)
    max_risk_pct: float | None = Field(default=None, gt=0, le=100)
    max_risk_amount: float | None = None
    risk_per_unit: float | None = None
    suggested_units: int | None = Field(default=None, ge=0)
    estimated_position_value: float | None = None


class TradeSignal(BaseModel):
    symbol: str
    action: SignalAction
    lifecycle: SignalLifecycle = SignalLifecycle.waiting
    generated_at: datetime
    timeframe: str
    current_price: float = Field(gt=0)
    entry_zone_low: float | None = Field(default=None, gt=0)
    entry_zone_high: float | None = Field(default=None, gt=0)
    ideal_entry: float | None = Field(default=None, gt=0)
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit_1: float | None = Field(default=None, gt=0)
    take_profit_2: float | None = Field(default=None, gt=0)
    take_profit_3: float | None = Field(default=None, gt=0)
    risk_reward: float | None = None
    signal_strength: float = Field(ge=0, le=100)
    scores: ScoreBreakdown
    technicals: TechnicalSnapshot
    thesis: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    invalidation: str | None = None
    position_plan: PositionPlan | None = None
    data_source: str

    @model_validator(mode="after")
    def validate_entry_zone(self) -> "TradeSignal":
        if self.entry_zone_low is not None and self.entry_zone_high is not None:
            if self.entry_zone_low > self.entry_zone_high:
                raise ValueError("entry_zone_low cannot exceed entry_zone_high")
        return self
