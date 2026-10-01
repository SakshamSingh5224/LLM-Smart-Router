from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_admin = Column(Boolean, default=False)

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    token_hash = Column(String, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked_at = Column(DateTime(timezone=True), nullable=True)

class Policy(Base):
    __tablename__ = "policies"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    tier_scope = Column(String, default="all")
    query_threshold = Column(Integer, nullable=True)
    token_threshold = Column(Integer, nullable=True)
    period = Column(String, default="monthly")
    action_on_exhaustion = Column(String, default="downgrade_to_low")

class UserPolicy(Base):
    __tablename__ = "user_policy"
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    policy_id = Column(Integer, ForeignKey("policies.id"))
    assigned_at = Column(DateTime(timezone=True), server_default=func.now())

class UsageLedger(Base):
    __tablename__ = "usage_ledger"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    ts = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    tier_used = Column(String, nullable=False)
    tokens_est = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    was_downgraded = Column(Boolean, default=False)
