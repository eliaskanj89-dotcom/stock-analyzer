from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from src.signals.models import SignalLifecycle, TradeSignal


@dataclass
class SignalHistoryStore:
    path: Path

    def __post_init__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, signal: TradeSignal) -> dict:
        record = signal.model_dump(mode="json")
        record["signal_id"] = f"{signal.symbol}-{int(signal.generated_at.timestamp())}"
        record["recorded_at"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        return record

    def list(self, symbol: str | None = None, limit: int = 100) -> list[dict]:
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        if symbol:
            records = [r for r in records if r.get("symbol") == symbol.upper()]
        return records[-max(1, min(limit, 1000)):][::-1]

    def performance(self, symbol: str | None = None) -> dict:
        records = self.list(symbol=symbol, limit=1000)
        terminal = {
            SignalLifecycle.tp1_hit.value,
            SignalLifecycle.tp2_hit.value,
            SignalLifecycle.tp3_hit.value,
            SignalLifecycle.stopped.value,
            SignalLifecycle.closed.value,
        }
        resolved = [r for r in records if r.get("lifecycle") in terminal]
        wins = [r for r in resolved if r.get("lifecycle") in {
            SignalLifecycle.tp1_hit.value, SignalLifecycle.tp2_hit.value, SignalLifecycle.tp3_hit.value
        }]
        return {
            "signals": len(records),
            "resolved": len(resolved),
            "wins": len(wins),
            "win_rate_pct": round(len(wins) / len(resolved) * 100, 2) if resolved else None,
            "note": "Performance is only reported for recorded resolved signals; unresolved signals are excluded.",
        }
