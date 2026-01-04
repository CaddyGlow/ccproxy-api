from __future__ import annotations

import time
from datetime import datetime as dt
from typing import Any

from sqlmodel import Session, col, func, select

from .models import AccessLog


class AnalyticsService:
    """Encapsulates analytics queries over the AccessLog table."""

    def __init__(self, engine: Any):
        self._engine = engine

    def query_logs(
        self,
        limit: int = 1000,
        start_time: float | None = None,
        end_time: float | None = None,
        model: str | None = None,
        service_type: str | None = None,
        cursor: float | None = None,
        order: str = "desc",
    ) -> dict[str, Any]:
        with Session(self._engine) as session:
            statement = select(AccessLog)

            start_dt = dt.fromtimestamp(start_time) if start_time else None
            end_dt = dt.fromtimestamp(end_time) if end_time else None
            cursor_dt = dt.fromtimestamp(cursor) if cursor else None

            if start_dt:
                statement = statement.where(AccessLog.timestamp >= start_dt)
            if end_dt:
                statement = statement.where(AccessLog.timestamp <= end_dt)
            if model:
                statement = statement.where(AccessLog.model == model)
            if service_type:
                statement = statement.where(AccessLog.service_type == service_type)

            # Cursor-based pagination using timestamp
            # For descending order (newest first): use timestamp < cursor
            # For ascending order (oldest first): use timestamp > cursor
            if cursor_dt:
                if order.lower() == "asc":
                    statement = statement.where(AccessLog.timestamp > cursor_dt)
                else:
                    statement = statement.where(AccessLog.timestamp < cursor_dt)

            if order.lower() == "asc":
                statement = statement.order_by(col(AccessLog.timestamp).asc()).limit(
                    limit
                )
            else:
                statement = statement.order_by(col(AccessLog.timestamp).desc()).limit(
                    limit
                )
            results = session.exec(statement).all()
            payload = [log.model_dump() for log in results]

            # Compute next cursor from last item in current page
            next_cursor = None
            if results:
                last = results[-1]
                next_cursor = last.timestamp.timestamp()

            return {
                "results": payload,
                "limit": limit,
                "count": len(results),
                "order": order.lower(),
                "cursor": cursor,
                "next_cursor": next_cursor,
                "has_more": len(results) == limit,
                "query_time": time.time(),
                "backend": "sqlmodel",
            }

    def get_analytics(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        model: str | None = None,
        service_type: str | None = None,
        hours: int | None = 24,
    ) -> dict[str, Any]:
        if start_time is None and end_time is None and hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        start_dt = dt.fromtimestamp(start_time) if start_time else None
        end_dt = dt.fromtimestamp(end_time) if end_time else None

        def build_filters() -> list[Any]:
            conditions: list[Any] = []
            if start_dt:
                conditions.append(AccessLog.timestamp >= start_dt)
            if end_dt:
                conditions.append(AccessLog.timestamp <= end_dt)
            if model:
                conditions.append(AccessLog.model == model)
            if service_type:
                parts = [s.strip() for s in service_type.split(",")]
                include = [p for p in parts if not p.startswith("!")]
                exclude = [p[1:] for p in parts if p.startswith("!")]
                if include:
                    conditions.append(col(AccessLog.service_type).in_(include))
                if exclude:
                    conditions.append(~col(AccessLog.service_type).in_(exclude))
            return conditions

        with Session(self._engine) as session:
            filters = build_filters()

            total_requests = session.exec(
                select(func.count()).select_from(AccessLog).where(*filters)
            ).first()
            total_successful_requests = session.exec(
                select(func.count())
                .select_from(AccessLog)
                .where(
                    *filters, AccessLog.status_code >= 200, AccessLog.status_code < 400
                )
            ).first()
            total_error_requests = session.exec(
                select(func.count())
                .select_from(AccessLog)
                .where(*filters, AccessLog.status_code >= 400)
            ).first()
            avg_duration = session.exec(
                select(func.avg(AccessLog.duration_ms))
                .select_from(AccessLog)
                .where(*filters)
            ).first()
            total_cost = session.exec(
                select(func.sum(AccessLog.cost_usd))
                .select_from(AccessLog)
                .where(*filters)
            ).first()
            total_tokens_input = session.exec(
                select(func.sum(AccessLog.tokens_input))
                .select_from(AccessLog)
                .where(*filters)
            ).first()
            total_tokens_output = session.exec(
                select(func.sum(AccessLog.tokens_output))
                .select_from(AccessLog)
                .where(*filters)
            ).first()
            total_cache_read_tokens = session.exec(
                select(func.sum(AccessLog.cache_read_tokens))
                .select_from(AccessLog)
                .where(*filters)
            ).first()
            total_cache_write_tokens = session.exec(
                select(func.sum(AccessLog.cache_write_tokens))
                .select_from(AccessLog)
                .where(*filters)
            ).first()

            services = session.exec(
                select(AccessLog.service_type).distinct().where(*filters)
            ).all()
            breakdown: dict[str, Any] = {}
            for svc in services:
                svc_filters = filters + [AccessLog.service_type == svc]
                svc_count = session.exec(
                    select(func.count()).select_from(AccessLog).where(*svc_filters)
                ).first()
                svc_success = session.exec(
                    select(func.count())
                    .select_from(AccessLog)
                    .where(
                        *svc_filters,
                        AccessLog.status_code >= 200,
                        AccessLog.status_code < 400,
                    )
                ).first()
                svc_error = session.exec(
                    select(func.count())
                    .select_from(AccessLog)
                    .where(*svc_filters, AccessLog.status_code >= 400)
                ).first()
                svc_avg = session.exec(
                    select(func.avg(AccessLog.duration_ms))
                    .select_from(AccessLog)
                    .where(*svc_filters)
                ).first()
                svc_cost = session.exec(
                    select(func.sum(AccessLog.cost_usd))
                    .select_from(AccessLog)
                    .where(*svc_filters)
                ).first()
                svc_in = session.exec(
                    select(func.sum(AccessLog.tokens_input))
                    .select_from(AccessLog)
                    .where(*svc_filters)
                ).first()
                svc_out = session.exec(
                    select(func.sum(AccessLog.tokens_output))
                    .select_from(AccessLog)
                    .where(*svc_filters)
                ).first()
                svc_cr = session.exec(
                    select(func.sum(AccessLog.cache_read_tokens))
                    .select_from(AccessLog)
                    .where(*svc_filters)
                ).first()
                svc_cw = session.exec(
                    select(func.sum(AccessLog.cache_write_tokens))
                    .select_from(AccessLog)
                    .where(*svc_filters)
                ).first()

                breakdown[str(svc)] = {
                    "request_count": svc_count or 0,
                    "successful_requests": svc_success or 0,
                    "error_requests": svc_error or 0,
                    "success_rate": (svc_success or 0) / (svc_count or 1) * 100
                    if svc_count
                    else 0,
                    "error_rate": (svc_error or 0) / (svc_count or 1) * 100
                    if svc_count
                    else 0,
                    "avg_duration_ms": svc_avg or 0,
                    "total_cost_usd": svc_cost or 0,
                    "total_tokens_input": svc_in or 0,
                    "total_tokens_output": svc_out or 0,
                    "total_cache_read_tokens": svc_cr or 0,
                    "total_cache_write_tokens": svc_cw or 0,
                    "total_tokens_all": (svc_in or 0)
                    + (svc_out or 0)
                    + (svc_cr or 0)
                    + (svc_cw or 0),
                }

            return {
                "summary": {
                    "total_requests": total_requests or 0,
                    "total_successful_requests": total_successful_requests or 0,
                    "total_error_requests": total_error_requests or 0,
                    "avg_duration_ms": avg_duration or 0,
                    "total_cost_usd": total_cost or 0,
                    "total_tokens_input": total_tokens_input or 0,
                    "total_tokens_output": total_tokens_output or 0,
                    "total_cache_read_tokens": total_cache_read_tokens or 0,
                    "total_cache_write_tokens": total_cache_write_tokens or 0,
                    "total_tokens_all": (total_tokens_input or 0)
                    + (total_tokens_output or 0)
                    + (total_cache_read_tokens or 0)
                    + (total_cache_write_tokens or 0),
                },
                "token_analytics": {
                    "input_tokens": total_tokens_input or 0,
                    "output_tokens": total_tokens_output or 0,
                    "cache_read_tokens": total_cache_read_tokens or 0,
                    "cache_write_tokens": total_cache_write_tokens or 0,
                    "total_tokens": (total_tokens_input or 0)
                    + (total_tokens_output or 0)
                    + (total_cache_read_tokens or 0)
                    + (total_cache_write_tokens or 0),
                },
                "request_analytics": {
                    "total_requests": total_requests or 0,
                    "successful_requests": total_successful_requests or 0,
                    "error_requests": total_error_requests or 0,
                    "success_rate": (total_successful_requests or 0)
                    / (total_requests or 1)
                    * 100
                    if total_requests
                    else 0,
                    "error_rate": (total_error_requests or 0)
                    / (total_requests or 1)
                    * 100
                    if total_requests
                    else 0,
                },
                "service_type_breakdown": breakdown,
                "query_time": time.time(),
                "backend": "sqlmodel",
            }
