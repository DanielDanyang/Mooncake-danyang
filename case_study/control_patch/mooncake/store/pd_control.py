# SPDX-License-Identifier: Apache-2.0
"""Process-local coordination between PD writes and Store puts.

This is an experimental control hook for Mooncake Store+PD contention studies.
It deliberately stays process-local because the current prefill-side
MooncakeConnector and MooncakeStoreConnector live in the same engine process.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass
class WaitResult:
    waited_s: float
    reason: str


class PDControlState:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._pending = 0
        self._active = 0
        self._active_bytes = 0
        self._last_transfer_id: str | None = None

    @property
    def mode(self) -> str:
        return os.getenv("MOONCAKE_PD_PROTECT_MODE", "off")

    @property
    def grace_s(self) -> float:
        return _env_float("MOONCAKE_PD_PROTECT_GRACE_MS", 10.0) / 1000.0

    @property
    def max_wait_s(self) -> float:
        return _env_float("MOONCAKE_PD_PROTECT_MAX_WAIT_MS", 500.0) / 1000.0

    @property
    def min_bytes(self) -> int:
        return _env_int("MOONCAKE_PD_PROTECT_MIN_BYTES", 1)

    def enabled(self) -> bool:
        return self.mode != "off"

    def mark_pending(self, total_bytes: int, transfer_id: str | None = None) -> None:
        if total_bytes < self.min_bytes:
            return
        with self._cond:
            if self._pending == 0 and self._active == 0:
                self._pending = 1
            self._last_transfer_id = transfer_id
            self._cond.notify_all()

    def begin_pd(self, total_bytes: int, transfer_id: str | None = None) -> None:
        if total_bytes < self.min_bytes:
            return
        with self._cond:
            if self._pending:
                self._pending -= 1
            self._active += 1
            self._active_bytes += total_bytes
            self._last_transfer_id = transfer_id
            self._cond.notify_all()

    def end_pd(self, total_bytes: int, transfer_id: str | None = None) -> None:
        if total_bytes < self.min_bytes:
            return
        with self._cond:
            if self._active:
                self._active -= 1
            self._active_bytes = max(0, self._active_bytes - total_bytes)
            self._last_transfer_id = transfer_id
            self._cond.notify_all()

    def clear_pending(self, total_bytes: int = 0, transfer_id: str | None = None) -> None:
        with self._cond:
            if self._pending:
                self._pending -= 1
            self._last_transfer_id = transfer_id
            self._cond.notify_all()

    def snapshot(self) -> dict[str, int | str | None]:
        with self._cond:
            return {
                "pending": self._pending,
                "active": self._active,
                "active_bytes": self._active_bytes,
                "transfer_id": self._last_transfer_id,
            }

    def wait_for_pd_quiet(self) -> WaitResult:
        """Wait for pending/active PD to drain.

        Store often reaches its put call a few milliseconds before
        request_finished() marks the PD transfer ready.  The grace window gives
        that explicit signal a chance to arrive; once pending/active is seen,
        Store waits until it drains or max_wait expires.
        """
        if not self.enabled():
            return WaitResult(0.0, "disabled")

        start = time.perf_counter()
        with self._cond:
            if self._pending == 0 and self._active == 0 and self.grace_s > 0:
                self._cond.wait(timeout=self.grace_s)

            if self._pending == 0 and self._active == 0:
                return WaitResult(time.perf_counter() - start, "no_pd")

            deadline = start + self.max_wait_s
            while self._pending or self._active:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return WaitResult(time.perf_counter() - start, "timeout")
                self._cond.wait(timeout=remaining)

        return WaitResult(time.perf_counter() - start, "pd_done")


pd_control = PDControlState()
