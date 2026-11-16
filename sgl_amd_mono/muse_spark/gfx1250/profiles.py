# SPDX-License-Identifier: Apache-2.0
"""Compile-time launch profiles for the gfx1250 Muse Spark TP4 shard."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    resident_blocks: int


PROFILES = {
    "cu96_tp4": Profile("cu96_tp4", 96),
    "cu192_tp4": Profile("cu192_tp4", 192),
}
DEFAULT_PROFILE = "cu192_tp4"
PROFILE_ENV = "SGLANG_MUSE_GFX1250_PROFILE"


def active_profile() -> Profile:
    name = os.environ.get(PROFILE_ENV, DEFAULT_PROFILE)
    try:
        return PROFILES[name]
    except KeyError as error:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"{PROFILE_ENV}={name!r}; choose one of: {choices}") from error


ACTIVE_PROFILE = active_profile()
