"""SQLAlchemy database models for metrics storage in Claude Code Proxy API Server."""

import json
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func


Base: Any = declarative_base()


class MetricsSnapshot(Base):
    """Table for storing metrics snapshots."""

    __tablename__ = "metrics_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    metric_name = Column(String(255), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)  # gauge, counter, histogram
    labels = Column(Text, nullable=False)  # JSON serialized labels
    value = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=func.now())

    __table_args__ = (
        Index("idx_metrics_timestamp_name", "timestamp", "metric_name"),
        Index("idx_metrics_type_name", "metric_type", "metric_name"),
    )


class RequestLog(Base):
    """Table for storing detailed request logs."""

    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    method = Column(String(10), nullable=False)
    endpoint = Column(String(255), nullable=False, index=True)
    api_type = Column(String(50), nullable=False, index=True)  # anthropic, openai
    model = Column(String(255), nullable=True, index=True)
    status_code = Column(Integer, nullable=False, index=True)
    duration_ms = Column(Float, nullable=False)
    request_size = Column(Integer, nullable=False, default=0)
    response_size = Column(Integer, nullable=False, default=0)
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    cost_dollars = Column(Float, nullable=False, default=0.0)
    user_agent = Column(Text, nullable=True)
    user_agent_category = Column(String(50), nullable=False, index=True)
    error_type = Column(String(100), nullable=True, index=True)
    created_at = Column(DateTime, nullable=False, default=func.now())

    __table_args__ = (
        Index("idx_request_logs_timestamp_endpoint", "timestamp", "endpoint"),
        Index("idx_request_logs_api_type_model", "api_type", "model"),
        Index("idx_request_logs_status_error", "status_code", "error_type"),
        Index(
            "idx_request_logs_daily_agg", "timestamp", "endpoint", "api_type", "model"
        ),
    )


class DailyAggregate(Base):
    """Table for storing daily aggregated metrics."""

    __tablename__ = "daily_aggregates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, index=True)
    endpoint = Column(String(255), nullable=False, index=True)
    api_type = Column(String(50), nullable=False, index=True)
    model = Column(String(255), nullable=True, index=True)
    total_requests = Column(Integer, nullable=False, default=0)
    total_errors = Column(Integer, nullable=False, default=0)
    avg_duration_ms = Column(Float, nullable=False, default=0.0)
    p95_duration_ms = Column(Float, nullable=False, default=0.0)
    p99_duration_ms = Column(Float, nullable=False, default=0.0)
    total_input_tokens = Column(Integer, nullable=False, default=0)
    total_output_tokens = Column(Integer, nullable=False, default=0)
    total_cost_dollars = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=func.now())

    __table_args__ = (
        Index("idx_daily_agg_date_endpoint", "date", "endpoint"),
        Index("idx_daily_agg_api_type_model", "api_type", "model"),
        Index(
            "idx_daily_agg_unique", "date", "endpoint", "api_type", "model", unique=True
        ),
    )


def create_database_engine(database_url: str = "sqlite:///metrics.db") -> Any:
    """Create and return SQLAlchemy engine.

    Args:
        database_url: Database connection URL

    Returns:
        SQLAlchemy engine instance
    """
    engine = create_engine(
        database_url,
        echo=False,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
    )
    return engine


def create_session_factory(engine: Any) -> Any:
    """Create and return SQLAlchemy session factory.

    Args:
        engine: SQLAlchemy engine instance

    Returns:
        SQLAlchemy session factory
    """
    return sessionmaker(bind=engine)


def create_tables(engine: Any) -> None:
    """Create all database tables.

    Args:
        engine: SQLAlchemy engine instance
    """
    Base.metadata.create_all(engine)


def serialize_labels(labels: dict[str, Any]) -> str:
    """Serialize labels dictionary to JSON string.

    Args:
        labels: Labels dictionary

    Returns:
        JSON string representation of labels
    """
    return json.dumps(labels, sort_keys=True)


def deserialize_labels(labels_json: str) -> dict[str, Any]:
    """Deserialize labels JSON string to dictionary.

    Args:
        labels_json: JSON string representation of labels

    Returns:
        Labels dictionary
    """
    return json.loads(labels_json)  # type: ignore[no-any-return]
