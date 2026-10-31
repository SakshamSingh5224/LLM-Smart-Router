import datetime
import logging
from typing import Literal, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

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
        return (self.db.query(Policy)
                .join(UserPolicy, UserPolicy.policy_id == Policy.id)
                .filter(UserPolicy.user_id == user_id).first())

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
        # One SQL round trip for the policy and monthly usage. Do not cache
        # authorization/quota state or defer accounting to improve latency.
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        start_of_month = datetime.datetime(now.year, now.month, 1)
        row = (self.db.query(
                    Policy, func.count(UsageLedger.id).label("queries"),
                    func.coalesce(func.sum(UsageLedger.tokens_est), 0).label("tokens"))
               .join(UserPolicy, UserPolicy.policy_id == Policy.id)
               .outerjoin(UsageLedger, and_(
                   UsageLedger.user_id == UserPolicy.user_id,
                   UsageLedger.ts >= start_of_month,
                   or_(Policy.tier_scope.is_(None), Policy.tier_scope != "high", UsageLedger.tier_used == HIGH)))
               .filter(UserPolicy.user_id == user_id)
               .group_by(*Policy.__table__.columns).first())
        if row is None:
            return PolicyDecision(action="allow", reason="No policy assigned, fail-open default.")
        policy = row[0]
        if policy.action_on_exhaustion == "allow_overage":
            return PolicyDecision(action="allow_overage", reason="Admin/unlimited policy active.")
        usage = {"queries": row.queries or 0, "tokens": row.tokens or 0}

        exhausted = False
        if policy.query_threshold is not None and usage["queries"] >= policy.query_threshold:
            exhausted = True
        if policy.token_threshold is not None and usage["tokens"] >= policy.token_threshold:
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

