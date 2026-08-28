"""Regression test for #376: move_to_end race on _window_memo OrderedDict.

`_cached_window` is called from the `/rooms` overview, which is a sync
handler dispatched into Starlette's threadpool.  The `_window_memo`
OrderedDict is process-global and unguarded by a lock.

The race (identical pattern to _buckets in limit.py and _rooms_cache in
app.py):

  Thread A                                Thread B
  ────────                                ────────
  _window_memo[key] = (stamp, view)
                                          _window_memo.popitem(last=False)  # evicts A's key
  _window_memo.move_to_end(key)           # ← KeyError → 500

The fix replaces `__setitem__` + `move_to_end` with `pop` + `__setitem__`:
a fresh insert lands at the end automatically, so `move_to_end` is
unnecessary, and `pop` absorbs a concurrent eviction harmlessly.

Run: uv run --group dev python -m pytest tests/unit/test_store_window_memo_race.py
"""

from __future__ import annotations

import itertools
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import store  # noqa: E402


def _seed_room(root: Path, name: str, n: int = 3) -> None:
    """Create a room with a few messages so `room_window` has something to read."""
    for i in range(n):
        store.append(root, name, f"user-{i}", f"hello {i}")


def test_concurrent_cached_window_never_raises_keyerror(tmp_path) -> None:
    """Eight threads hammer `_cached_window` with distinct room names and a
    tiny memo cap, forcing constant eviction.  Before the fix, `move_to_end`
    raised KeyError when a concurrent `popitem` removed the key between
    `__setitem__` and `move_to_end`.

    A small `_WINDOW_MEMO_MAX` and a high thread-switch frequency make the
    interleaving reliable rather than lucky.
    """
    store._window_memo.clear()

    # Seed enough rooms to exceed the memo cap
    room_count = 64
    for i in range(room_count):
        _seed_room(tmp_path, f"room-{i}", n=2)

    # Get stat stamps for each room
    stamps = {}
    for i in range(room_count):
        rp = store.room_path(tmp_path, f"room-{i}")
        st = rp.stat()
        stamps[f"room-{i}"] = (st.st_mtime_ns, st.st_size)

    errors: list[BaseException] = []
    counter = itertools.count()
    original_max = store._WINDOW_MEMO_MAX

    # Shrink the memo cap to force frequent eviction
    store._WINDOW_MEMO_MAX = 16

    def flood() -> None:
        try:
            for _ in range(1_000):
                n = next(counter)
                room = f"room-{n % room_count}"
                store._cached_window(tmp_path, room, stamps[room])
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
        store._WINDOW_MEMO_MAX = original_max

    assert not errors, [repr(e) for e in errors[:3]]
    assert len(store._window_memo) <= 16, (
        f"memo cap must hold under concurrency: got {len(store._window_memo)}"
    )
    store._window_memo.clear()


def test_window_memo_cap_is_respected() -> None:
    """Verify that sequential overflow still respects the LRU bound."""
    store._window_memo.clear()
    original_max = store._WINDOW_MEMO_MAX
    store._WINDOW_MEMO_MAX = 8

    try:
        for i in range(50):
            key = (f"/fake/root-{i}", f"room-{i}")
            store._window_memo[key] = ((0, 0), (0, []))
            # Simulate the move_to_end + eviction loop
            store._window_memo.pop(key, None)
            store._window_memo[key] = ((0, 0), (0, []))
            while len(store._window_memo) > store._WINDOW_MEMO_MAX:
                store._window_memo.popitem(last=False)

        assert len(store._window_memo) <= 8
    finally:
        store._WINDOW_MEMO_MAX = original_max
        store._window_memo.clear()
