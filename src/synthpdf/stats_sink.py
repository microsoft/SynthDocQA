# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Per-document generation statistics sink.

Generators call ``log_artifact()`` to record structured per-artifact timing
data.  The CLI layer reads the log at the end of each document, combines it
with phase timings and pool-size counts, and writes a compact JSON stats
file.  ``clear()`` must be called before each new document to reset the
accumulated data.

A plain ``threading.Lock`` guards the module-level state so that concurrent
worker threads (used by the variant generators' internal thread pools) can
safely append to the same log while a document is being generated.  Documents
are generated one at a time by the outer CLI loop, so there is no
cross-document contamination.
"""
from __future__ import annotations

import threading

_lock          = threading.Lock()
_artifact_log: list[dict]       = []
_type_times:   dict[str, float] = {}


# ---------------------------------------------------------------------------
# Writers  (called by generators)
# ---------------------------------------------------------------------------

def log_artifact(**kwargs) -> None:
    """Append one per-artifact timing record.  Thread-safe."""
    with _lock:
        _artifact_log.append(dict(kwargs))


def set_type_times(d: dict[str, float]) -> None:
    """Store the per-type aggregate generation times.  Thread-safe."""
    global _type_times
    with _lock:
        _type_times = dict(d)


# ---------------------------------------------------------------------------
# Readers  (called by CLI after each document)
# ---------------------------------------------------------------------------

def get_log() -> list[dict]:
    """Return a snapshot of the current artifact log."""
    with _lock:
        return list(_artifact_log)


def get_type_times() -> dict[str, float]:
    """Return a snapshot of the per-type generation times."""
    with _lock:
        return dict(_type_times)


# ---------------------------------------------------------------------------
# Reset  (called by CLI before each document)
# ---------------------------------------------------------------------------

def clear() -> None:
    """Reset both the artifact log and type-times for the next document."""
    global _artifact_log, _type_times
    with _lock:
        _artifact_log = []
        _type_times   = {}
