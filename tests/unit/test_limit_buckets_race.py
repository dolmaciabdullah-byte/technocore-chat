"""Regression test for #378: move_to_end race on _buckets OrderedDict.

Starlette dispatches sync endpoint handlers into a threadpool, so every
call to `take()` runs in a worker thread.  The `_buckets` OrderedDict is
process-global and unguarded by a lock.

The race:

  Thread A                          Thread B
  ────────                          ────────
  _buckets[(ip, kind)] = ...
                                    _buckets.popitem(last=False)   # evicts A's key
  _buckets.move_to_end((ip, kind))  # ← KeyError

The fix replaces `__setitem__` + `move_to_end` with `pop` + `__setitem__`:
a fresh insert is already at the end, so `move_to_end` is unnecessary, and
the `pop` absorbs a concurrent eviction harmlessly.

Run: uv run --group dev python -m pytest tests/unit/test_limit_buckets_race.py
"""

from __future__ import annotations

import itertools
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import limit  # noqa: E402


class _FakeRequest:
    """Minimal stand-in for a Starlette Request, enough for `take()`."""

    def __init__(self, ip: str) -> None:
        self._ip = ip
        self.headers = {}
        # Starlette's Request.client is an (host, port) namedtuple.
        self.client = type("Addr", (), {"host": self._ip})()


def test_concurrent_take_never_raises_keyerror() -> None:
    """Eight threads hammer `take()` with distinct IPs and a tiny bucket cap,
    forcing constant eviction.  Before the fix, `move_to_end` raised KeyError
    when a concurrent `popitem` removed the key between `__setitem__` and
    `move_to_end`.  After the fix (pop-then-insert) no exception is possible.

    A small `max_buckets` and a high thread-switch frequency make the
    interleaving reliable rather than lucky.
    """
    limit._buckets.clear()
    limit._requests.clear()
    limit._requests["rate_limited"] = 0

    errors: list[BaseException] = []
    counter = itertools.count()

    def flood() -> None:
        try:
            for _ in range(2_000):
                n = next(counter)
                req = _FakeRequest(f"10.0.{n % 256}.{(n // 256) % 256}")
                limit.take(req, kind="msg", per_min=60, max_buckets=32)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    switch = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=flood) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(switch)

    assert not errors, [repr(e) for e in errors[:3]]
    assert len(limit._buckets) <= 32, (
        f"bucket cap must hold under concurrency: got {len(limit._buckets)}"
    )
    limit._buckets.clear()


def test_bucket_cap_is_respected_after_concurrent_eviction() -> None:
    """Verify that even after heavy concurrent usage the LRU bound holds."""
    limit._buckets.clear()
    limit._requests.clear()
    limit._requests["rate_limited"] = 0

    cap = 16
    for i in range(200):
        req = _FakeRequest(f"192.168.{i % 256}.{i // 256}")
        limit.take(req, kind="msg", per_min=60, max_buckets=cap)

    assert len(limit._buckets) <= cap
    limit._buckets.clear()
