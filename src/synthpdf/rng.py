# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. See LICENSE.txt in the project root.
"""Deterministic randomness utilities for the synthpdf package."""
from __future__ import annotations

import hashlib
import struct

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Create a seeded numpy random Generator."""
    return np.random.default_rng(seed)


def fork_rng(parent_rng: np.random.Generator, salt: str) -> np.random.Generator:
    """Fork a child RNG from a parent RNG using a deterministic, hash-based approach.

    The parent RNG provides a base integer (advancing its state by one step).
    That integer is hashed together with the salt string to produce a fully
    independent child seed, so child RNGs for different salts never share state.
    """
    base: int = int(parent_rng.integers(0, 2**63))
    raw = struct.pack(">Q", base) + salt.encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    child_seed = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(child_seed)
