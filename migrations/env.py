"""Alembic environment script.

Two things are deliberately NOT independent copies of what's in gateway/db/:

1. The database URL comes from gateway.db.database.SQLALCHEMY_DATABASE_URL -
   the exact same variable the running app uses - instead of being re-read
   from DATABASE_URL here too. Two separate "figure out the DB URL" code
   paths are exactly how an app and its migrations quietly point at two
   different databases; importing the app's own value makes that impossible.
2. target_metadata is gateway.db.database.Base.metadata, and we import
   gateway.db.models (for its side effect of registering every model class
   on that Base) before reading it - so `alembic revision --autogenerate`
   sees the real, current set of models, not a stale/partial list.
"""
from __future__ import annotations

import sys
from pathlib import Path
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the `gateway` package importable regardless of the cwd this is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gateway.db.database import Base, SQLALCHEMY_DATABASE_URL  # noqa: E402
import gateway.db.models  # noqa: E402,F401  (registers all models onto Base.metadata)

config = context.config
config.set_main_option("sqlalchemy.url", SQLALCHEMY_DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it (`alembic upgrade head --sql`)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """The normal path: connect to the real database and apply migrations."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
