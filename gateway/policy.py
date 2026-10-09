import datetime
import logging
from typing import Literal, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from gateway.metrics import USAGE_LEDGER_WRITE_FAILURES

log = logging.getLogger("gateway.policy")

from gateway.db.models import UserPolicy, Policy, UsageLedger
from smartrouter.labels import HIGH

class PolicyDecision(BaseModel):
    action: Literal["allow", "downgrade_to_low", "block", "allow_overage"]
    reason: str
    is_downgraded: bool = False

class PolicyEngine:
    def __init__(self, db: Session):
        self.db = db

    def get_user_policy(self, user_id: int) -> Optional[Policy]:
        up = self.db.query(UserPolicy).filter(UserPolicy.user_id == user_id).first()
        if not up:
            return None
        return self.db.query(Policy).filter(Policy.id == up.policy_id).first()

    def get_current_usage(self, user_id: int, tier_scope: str = "all") -> dict:
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        start_of_month = datetime.datetime(now.year, now.month, 1)
        
        query = self.db.query(
            func.count(UsageLedger.id).label("queries"),
            func.sum(UsageLedger.tokens_est).label("tokens")
        ).filter(
            UsageLedger.user_id == user_id,
            UsageLedger.ts >= start_of_month
        )
        
        if tier_scope == "high":
            query = query.filter(UsageLedger.tier_used == HIGH)
            
        res = query.first()
        return {
            "queries": res.queries or 0,
            "tokens": res.tokens or 0
        }

    def evaluate(self, user_id: int) -> PolicyDecision:
        policy = self.get_user_policy(user_id)
        if not policy:
            return PolicyDecision(action="allow", reason="No policy assigned, fail-open default.")

        if policy.action_on_exhaustion == "allow_overage":
            return PolicyDecision(action="allow_overage", reason="Admin/unlimited policy active.")

        usage = self.get_current_usage(user_id, policy.tier_scope)
        
        exhausted = False
        if policy.query_threshold and usage["queries"] >= policy.query_threshold:
            exhausted = True
        if policy.token_threshold and usage["tokens"] >= policy.token_threshold:
            exhausted = True

        if exhausted:
            if policy.action_on_exhaustion == "block":
                return PolicyDecision(action="block", reason="Monthly quota exhausted.")
            else:
                return PolicyDecision(
                    action="downgrade_to_low", 
                    reason="Monthly quota exhausted. Downgrading to low-cost tier.", 
                    is_downgraded=True
                )
        
        return PolicyDecision(action="allow", reason="Within quota.")

    def record_usage(self, user_id: int, tier: str, tokens: int, cost: float, downgraded: bool):
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        record = UsageLedger(
            user_id=user_id,
            ts=now,
            tier_used=tier,
            tokens_est=tokens,
            cost_usd=cost,
            was_downgraded=downgraded
        )
        try:
            self.db.add(record)
            self.db.commit()
            return True
        except Exception:
            # Usage is best-effort telemetry; a transient DB failure must not
            # turn a successful model response into a 500 or kill an SSE stream.
            try:
                self.db.rollback()
            except Exception:
                log.exception("Rollback after usage-ledger failure also failed.")
            USAGE_LEDGER_WRITE_FAILURES.inc()
            log.exception("Usage-ledger write failed; continuing without recording usage.")
            return False
