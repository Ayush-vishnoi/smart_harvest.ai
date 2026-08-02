"""Deployment-safe persistence for predictions and chatbot activity."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, desc, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, scoped_session, sessionmaker

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    state: Mapped[str | None] = mapped_column(String(100), index=True)
    crop: Mapped[str | None] = mapped_column(String(100), index=True)
    result_label: Mapped[str | None] = mapped_column(String(255))
    result_value: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    request_data: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    response_data: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )


class ChatInteraction(Base):
    __tablename__ = "chat_interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    reply: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )


class Database:
    """Small SQLAlchemy service that supports local SQLite and hosted PostgreSQL."""

    def __init__(self) -> None:
        self.engine = None
        self.sessions = None
        self.url = ""

    @staticmethod
    def _database_url(instance_path: str) -> str:
        configured = os.environ.get("DATABASE_URL", "").strip()
        if configured.startswith("postgres://"):
            configured = "postgresql+psycopg://" + configured[len("postgres://") :]
        elif configured.startswith("postgresql://"):
            configured = "postgresql+psycopg://" + configured[len("postgresql://") :]
        if configured:
            return configured

        Path(instance_path).mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{Path(instance_path) / 'smart_harvest.db'}"

    def init_app(self, app) -> None:
        self.url = self._database_url(app.instance_path)
        options: dict[str, Any] = {"pool_pre_ping": True}
        if self.url.startswith("sqlite"):
            options["connect_args"] = {"check_same_thread": False}

        self.engine = create_engine(self.url, **options)
        self.sessions = scoped_session(
            sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)
        )
        app.teardown_appcontext(lambda _error=None: self.sessions.remove())

        try:
            Base.metadata.create_all(self.engine)
            app.config["DATABASE_AVAILABLE"] = True
            log.info("Database initialized using %s", self.backend_name)
        except Exception:
            app.config["DATABASE_AVAILABLE"] = False
            log.exception("Database initialization failed; persistence is disabled")

    @property
    def backend_name(self) -> str:
        return self.engine.url.get_backend_name() if self.engine is not None else "unconfigured"

    def is_healthy(self) -> bool:
        if self.engine is None:
            return False
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:
            log.warning("Database health check failed", exc_info=True)
            return False

    def save_prediction(
        self,
        prediction_type: str,
        request_data: dict[str, Any],
        response_data: dict[str, Any],
        *,
        result_label: str | None = None,
        result_value: float | None = None,
        confidence: float | None = None,
    ) -> None:
        if self.sessions is None:
            return
        session = self.sessions()
        try:
            session.add(
                Prediction(
                    prediction_type=prediction_type,
                    state=_optional_text(request_data.get("state")),
                    crop=_optional_text(request_data.get("crop")),
                    result_label=result_label,
                    result_value=result_value,
                    confidence=confidence,
                    request_data=json.dumps(request_data, default=str),
                    response_data=json.dumps(response_data, default=str),
                )
            )
            session.commit()
        except Exception:
            session.rollback()
            log.exception("Failed to persist %s prediction", prediction_type)

    def save_chat(self, message: str, reply: str, sources: list[str]) -> None:
        if self.sessions is None:
            return
        session = self.sessions()
        try:
            session.add(ChatInteraction(message=message, reply=reply, sources=json.dumps(sources)))
            session.commit()
        except Exception:
            session.rollback()
            log.exception("Failed to persist chat interaction")

    def recent_predictions(self, limit: int = 10) -> list[dict[str, Any]]:
        if self.sessions is None:
            return []
        session = self.sessions()
        try:
            rows = session.scalars(
                select(Prediction).order_by(desc(Prediction.created_at)).limit(max(1, min(limit, 100)))
            ).all()
            return [
                {
                    "id": row.id,
                    "type": row.prediction_type,
                    "state": row.state,
                    "crop": row.crop,
                    "label": row.result_label,
                    "value": row.result_value,
                    "confidence": row.confidence,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]
        except Exception:
            log.exception("Failed to load recent predictions")
            return []


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned[:255] if cleaned else None


database = Database()
