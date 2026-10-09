import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool

# Uses Neon's pooled connection string from environment variables
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db")

_connect_args = {}
_engine_kwargs = {}
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}
    if ":memory:" in SQLALCHEMY_DATABASE_URL:
        # Without StaticPool, every new connection to sqlite:///:memory: opens its
        # own blank, throwaway database, so tables created at startup silently
        # vanish on the next connection (e.g. a streaming response's background
        # task) -> "no such table: ..." errors. StaticPool shares one connection.
        _engine_kwargs["poolclass"] = StaticPool
else:
    # Neon/serverless Postgres may close idle pooled connections. Validate each
    # checkout and recycle connections before long idle periods.
    _engine_kwargs.update(pool_pre_ping=True, pool_recycle=300)

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=_connect_args, **_engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
