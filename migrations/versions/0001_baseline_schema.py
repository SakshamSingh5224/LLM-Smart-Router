"""baseline schema: users, refresh_tokens, policies, user_policy, usage_ledger

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-10-04 00:00:00

This migration represents the schema as it already exists today (created via
Base.metadata.create_all() before Alembic was introduced), including the
usage_ledger(user_id, ts) composite index fix. It is written by hand rather
than via `alembic revision --autogenerate` because there's no live DB
connection available to generate it against - but every column/type/
constraint below is a direct, deliberate match of gateway/db/models.py as it
stands right now. If you add or change a model going forward, generate the
*next* migration normally with --autogenerate; only this first one is
hand-written.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("is_admin", sa.Boolean(), server_default=sa.false()),
    )
    op.create_index("ix_users_id", "users", ["id"])
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id")),
        sa.Column("token_hash", sa.String(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_refresh_tokens_id", "refresh_tokens", ["id"])

    op.create_table(
        "policies",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("tier_scope", sa.String(), server_default="all"),
        sa.Column("query_threshold", sa.Integer(), nullable=True),
        sa.Column("token_threshold", sa.Integer(), nullable=True),
        sa.Column("period", sa.String(), server_default="monthly"),
        sa.Column("action_on_exhaustion", sa.String(), server_default="downgrade_to_low"),
    )
    op.create_index("ix_policies_id", "policies", ["id"])
    op.create_index("ix_policies_name", "policies", ["name"], unique=True)

    op.create_table(
        "user_policy",
        # NOTE: user_id alone is the primary key (not a (user_id, policy_id)
        # composite) - this mirrors models.py exactly: each user can have at
        # most one row here, i.e. one "current policy", not a history.
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), primary_key=True),
        sa.Column("policy_id", sa.Integer(), sa.ForeignKey("policies.id")),
        sa.Column("assigned_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "usage_ledger",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id")),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("tier_used", sa.String(), nullable=False),
        sa.Column("tokens_est", sa.Integer(), server_default="0"),
        sa.Column("cost_usd", sa.Float(), server_default="0.0"),
        sa.Column("was_downgraded", sa.Boolean(), server_default=sa.false()),
    )
    op.create_index("ix_usage_ledger_id", "usage_ledger", ["id"])
    op.create_index("ix_usage_ledger_user_id_ts", "usage_ledger", ["user_id", "ts"])


def downgrade() -> None:
    # Reverse dependency order: tables with FKs to others drop first.
    op.drop_table("usage_ledger")
    op.drop_table("user_policy")
    op.drop_table("policies")
    op.drop_table("refresh_tokens")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_id", table_name="users")
    op.drop_table("users")
