"""Phiên Postgres thử lại tối đa 3 lần khi mất kết nối."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from psycopg import Connection, sql
from psycopg import errors as pg_errors

log = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3


class PgSession:
    """Giữ một connection. Lỗi dữ liệu không thử lại."""

    def __init__(self, connect: Callable[[], Connection]) -> None:
        self._connect = connect
        self._connection: Connection | None = None

    def run(
        self,
        statement: sql.Composable | str,
        params: tuple[object, ...] | None,
        description: str,
        fetch: str,
    ) -> list[tuple[Any, ...]] | tuple[Any, ...] | None:
        holder: dict[str, list[tuple[Any, ...]] | tuple[Any, ...] | None] = {}

        def action(cursor: Any) -> None:
            cursor.execute(statement, params)
            if fetch == "all":
                holder["rows"] = cursor.fetchall()
            elif fetch == "one":
                holder["rows"] = cursor.fetchone()

        self._attempt(description, action)
        return holder.get("rows")

    def executemany(
        self,
        statement: sql.Composable,
        rows: Sequence[tuple[object, ...]],
        description: str,
    ) -> None:
        self._attempt(description, lambda cursor: cursor.executemany(statement, rows))

    def _attempt(self, description: str, action: Callable[[Any], None]) -> None:
        last: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                connection = self._alive()
                with connection.cursor() as cursor:
                    action(cursor)
                connection.commit()
                return
            except (pg_errors.OperationalError, pg_errors.InterfaceError) as exc:
                self._discard()
                last = exc
                log.warning(
                    "Postgres lỗi kết nối (%s) lần %d/%d: %s",
                    description,
                    attempt,
                    _MAX_ATTEMPTS,
                    exc,
                )
            except (pg_errors.DataError, pg_errors.ProgrammingError) as exc:
                self._rollback()
                raise ValueError(
                    f"Postgres từ chối '{description}': {exc}. "
                    "Sửa config.yaml hoặc chạy lại ingest với --recreate."
                ) from exc
        raise RuntimeError(
            f"Postgres lỗi kết nối ({description}) sau {_MAX_ATTEMPTS} lần: {last}. "
            "Kiểm tra `docker compose up -d pgvector` và pgvector.dsn trong config.yaml."
        )

    def _alive(self) -> Connection:
        connection = self._connection
        if connection is not None and not connection.closed:
            return connection
        opened = self._connect()
        self._connection = opened
        return opened

    def _rollback(self) -> None:
        connection = self._connection
        if connection is not None and not connection.closed:
            connection.rollback()

    def _discard(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None and not connection.closed:
            connection.close()
