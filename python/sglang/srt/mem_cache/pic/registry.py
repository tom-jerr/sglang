from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from sglang.srt.mem_cache.pic.types import (
    DependencyMode,
    PicSpanIdentity,
    PicSpanPayload,
    PicSpanRecord,
    SeamMetadata,
)
from typing_extensions import Self


@dataclass(slots=True)
class _Entry:
    record: PicSpanRecord
    readers: int = 0
    accepting_readers: bool = True
    on_retire: Callable[[PicSpanPayload], None] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RegistrationResult:
    record: PicSpanRecord
    inserted: bool


class PicSpanLease:
    """Reader lease that keeps a canonical physical payload alive."""

    __slots__ = ("_registry", "_released", "delta", "record", "target_position")

    def __init__(
        self,
        registry: PicSpanRegistry,
        record: PicSpanRecord,
        target_position: int,
    ):
        self._registry = registry
        self.record = record
        self.target_position = target_position
        self.delta = target_position - record.payload.source_position
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._registry._release(self.record.handle)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


class PicSpanRegistry:
    """Thread-safe collision-safe canonical span registry.

    Retirement is two-phase: the entry is first removed from lookup so no new
    request can acquire it, then its physical payload callback runs after the
    last existing lease releases.  This is the contract needed to compose PIC
    with radix eviction and asynchronous HiCache/PD transfers.
    """

    def __init__(self, *, max_entries: int = 100_000):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._buckets: dict[PicSpanIdentity, list[int]] = {}
        self._entries: dict[int, _Entry] = {}
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._next_handle = 1

    def register(
        self,
        *,
        identity: PicSpanIdentity,
        token_ids: tuple[int, ...],
        dependency_mode: DependencyMode,
        seam: SeamMetadata,
        payload: PicSpanPayload,
        on_retire: Callable[[PicSpanPayload], None] | None = None,
    ) -> RegistrationResult:
        if len(token_ids) != identity.token_count:
            raise ValueError("token_ids length does not match PIC identity")

        retire_callbacks: list[tuple[Callable[[PicSpanPayload], None], PicSpanPayload]]
        with self._lock:
            for handle in self._buckets.get(identity, ()):
                entry = self._entries[handle]
                if entry.accepting_readers and entry.record.token_ids == token_ids:
                    self._lru.move_to_end(handle)
                    return RegistrationResult(record=entry.record, inserted=False)

            handle = self._next_handle
            self._next_handle += 1
            record = PicSpanRecord(
                handle=handle,
                identity=identity,
                token_ids=token_ids,
                dependency_mode=dependency_mode,
                seam=seam,
                payload=payload,
            )
            self._entries[handle] = _Entry(record=record, on_retire=on_retire)
            self._buckets.setdefault(identity, []).append(handle)
            self._lru[handle] = None
            retire_callbacks = self._trim_locked()

        self._run_retire_callbacks(retire_callbacks)
        return RegistrationResult(record=record, inserted=True)

    def probe(
        self, identity: PicSpanIdentity, token_ids: tuple[int, ...]
    ) -> PicSpanRecord | None:
        """Observer-only lookup.  Live consumers must use `acquire`."""

        with self._lock:
            for handle in self._buckets.get(identity, ()):
                entry = self._entries[handle]
                if entry.accepting_readers and entry.record.token_ids == token_ids:
                    self._lru.move_to_end(handle)
                    return entry.record
        return None

    def acquire(
        self,
        identity: PicSpanIdentity,
        token_ids: tuple[int, ...],
        *,
        target_position: int,
    ) -> PicSpanLease | None:
        with self._lock:
            for handle in self._buckets.get(identity, ()):
                entry = self._entries[handle]
                if entry.accepting_readers and entry.record.token_ids == token_ids:
                    entry.readers += 1
                    self._lru.move_to_end(handle)
                    return PicSpanLease(self, entry.record, target_position)
        return None

    def retire(self, handle: int) -> bool:
        callback = None
        with self._lock:
            entry = self._entries.get(handle)
            if entry is None or not entry.accepting_readers:
                return False
            callback = self._begin_retire_locked(handle)
        self._run_retire_callbacks([callback] if callback is not None else [])
        return True

    def _release(self, handle: int) -> None:
        callback = None
        with self._lock:
            entry = self._entries.get(handle)
            if entry is None or entry.readers <= 0:
                raise RuntimeError(f"invalid PIC lease release for handle {handle}")
            entry.readers -= 1
            if entry.readers == 0 and not entry.accepting_readers:
                callback = self._finalize_retire_locked(handle)
        self._run_retire_callbacks([callback] if callback is not None else [])

    def _trim_locked(
        self,
    ) -> list[tuple[Callable[[PicSpanPayload], None], PicSpanPayload]]:
        callbacks = []
        while len(self._lru) > self.max_entries:
            handle = next(iter(self._lru))
            callback = self._begin_retire_locked(handle)
            if callback is not None:
                callbacks.append(callback)
        return callbacks

    def _begin_retire_locked(
        self, handle: int
    ) -> tuple[Callable[[PicSpanPayload], None], PicSpanPayload] | None:
        entry = self._entries[handle]
        entry.accepting_readers = False
        self._lru.pop(handle, None)
        bucket = self._buckets[entry.record.identity]
        bucket.remove(handle)
        if not bucket:
            del self._buckets[entry.record.identity]
        if entry.readers == 0:
            return self._finalize_retire_locked(handle)
        return None

    def _finalize_retire_locked(
        self, handle: int
    ) -> tuple[Callable[[PicSpanPayload], None], PicSpanPayload] | None:
        entry = self._entries.pop(handle)
        if entry.on_retire is None:
            return None
        return entry.on_retire, entry.record.payload

    @staticmethod
    def _run_retire_callbacks(
        callbacks: list[tuple[Callable[[PicSpanPayload], None], PicSpanPayload]],
    ) -> None:
        for callback, payload in callbacks:
            callback(payload)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "active_entries": len(self._lru),
                "retiring_entries": len(self._entries) - len(self._lru),
                "active_readers": sum(
                    entry.readers for entry in self._entries.values()
                ),
            }
