from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import pandas as pd

from src.signals.models import SignalAction, SignalLifecycle, TradeSignal


@dataclass
class SignalHistoryStore:
    path: Path

    def __post_init__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _write_all(self, records: list[dict]) -> None:
        self.path.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in records) + ("\n" if records else ""), encoding="utf-8")

    def append(self, signal: TradeSignal) -> dict:
        record = signal.model_dump(mode="json")
        record["signal_id"] = f"{signal.symbol}-{int(signal.generated_at.timestamp())}"
        record["recorded_at"] = datetime.now(timezone.utc).isoformat()
        record["entry_triggered_at"] = None
        record["tp1_hit_at"] = None
        record["tp2_hit_at"] = None
        record["tp3_hit_at"] = None
        record["stop_hit_at"] = None
        record["closed_at"] = None
        record["exit_price"] = None
        record["realized_return_pct"] = None
        record["realized_r_multiple"] = None
        with self._lock:
            records = self._read_all()
            records.append(record)
            self._write_all(records)
        return record

    def list(self, symbol: str | None = None, limit: int = 100) -> list[dict]:
        with self._lock:
            records = self._read_all()
        if symbol:
            records = [r for r in records if r.get("symbol") == symbol.upper()]
        return records[-max(1, min(limit, 1000)):][::-1]

    def evaluate(self, provider, symbol: str | None = None) -> dict:
        """Advance open signal lifecycles using post-publication OHLC bars."""
        with self._lock:
            records = self._read_all()
            changed = 0
            for r in records:
                if symbol and r.get("symbol") != symbol.upper():
                    continue
                if r.get("lifecycle") in {SignalLifecycle.tp3_hit.value, SignalLifecycle.stopped.value, SignalLifecycle.closed.value, SignalLifecycle.invalidated.value}:
                    continue
                if not r.get("ideal_entry") or not r.get("stop_loss"):
                    continue
                try:
                    bars = provider.history(r["symbol"], period="1y", interval="1d")
                    generated = pd.Timestamp(r["generated_at"])
                    if generated.tzinfo is not None:
                        generated = generated.tz_convert(None)
                    idx = bars.index
                    if getattr(idx, "tz", None) is not None:
                        idx = idx.tz_localize(None)
                    bars = bars.loc[idx >= generated.normalize()]
                    if bars.empty:
                        continue
                    before = r.get("lifecycle")
                    self._advance(r, bars)
                    if r.get("lifecycle") != before:
                        changed += 1
                except Exception:
                    continue
            self._write_all(records)
        return {"evaluated": len(records), "changed": changed}

    @staticmethod
    def _advance(r: dict, bars: pd.DataFrame) -> None:
        long_side = r["action"] in {SignalAction.long.value, SignalAction.strong_long.value}
        entry=float(r["ideal_entry"]); stop=float(r["stop_loss"])
        tps=[r.get("take_profit_1"),r.get("take_profit_2"),r.get("take_profit_3")]
        risk=abs(entry-stop)
        triggered=r.get("entry_triggered_at") is not None
        for ts,row in bars.iterrows():
            lo,hi=float(row["low"]),float(row["high"])
            stamp=pd.Timestamp(ts).isoformat()
            if not triggered:
                zlo=r.get("entry_zone_low") or entry; zhi=r.get("entry_zone_high") or entry
                if lo <= float(zhi) and hi >= float(zlo):
                    triggered=True; r["entry_triggered_at"]=stamp; r["lifecycle"]=SignalLifecycle.triggered.value
                else:
                    continue
            # Conservative same-bar ordering: if stop and target both touch, count stop first.
            stop_hit = lo <= stop if long_side else hi >= stop
            if stop_hit:
                r["lifecycle"]=SignalLifecycle.stopped.value; r["stop_hit_at"]=stamp; r["closed_at"]=stamp; r["exit_price"]=stop
                ret=(stop/entry-1)*100*(1 if long_side else -1)
                r["realized_return_pct"]=round(ret,2); r["realized_r_multiple"]=-1.0
                return
            for n,tp in reversed(list(enumerate(tps,1))):
                if tp is None: continue
                hit = hi >= float(tp) if long_side else lo <= float(tp)
                if hit:
                    r[f"tp{n}_hit_at"]=r.get(f"tp{n}_hit_at") or stamp
                    r["lifecycle"]={1:SignalLifecycle.tp1_hit.value,2:SignalLifecycle.tp2_hit.value,3:SignalLifecycle.tp3_hit.value}[n]
                    if n==3:
                        r["closed_at"]=stamp; r["exit_price"]=float(tp)
                        ret=(float(tp)/entry-1)*100*(1 if long_side else -1)
                        r["realized_return_pct"]=round(ret,2); r["realized_r_multiple"]=round(abs(float(tp)-entry)/risk,2) if risk else None
                    break

    def performance(self, symbol: str | None = None) -> dict:
        records=self.list(symbol=symbol,limit=1000)
        resolved=[r for r in records if r.get("closed_at")]
        wins=[r for r in resolved if (r.get("realized_return_pct") or 0)>0]
        returns=[float(r["realized_return_pct"]) for r in resolved if r.get("realized_return_pct") is not None]
        rvals=[float(r["realized_r_multiple"]) for r in resolved if r.get("realized_r_multiple") is not None]
        return {
            "signals":len(records),"resolved":len(resolved),"wins":len(wins),
            "win_rate_pct":round(len(wins)/len(resolved)*100,2) if resolved else None,
            "avg_return_pct":round(sum(returns)/len(returns),2) if returns else None,
            "avg_r_multiple":round(sum(rvals)/len(rvals),2) if rvals else None,
            "note":"Only permanently recorded, resolved signals are included. Same-bar stop/target ambiguity is scored conservatively as a stop.",
        }
