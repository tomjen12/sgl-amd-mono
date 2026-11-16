# SPDX-License-Identifier: Apache-2.0
"""Muse Spark MI450MC shapes for gfx1250: one SPX rank at TP4 / EP4.

This module contains compile-time shape facts only.  Kernel launch geometry and
LDS layouts live in :mod:`layout`, so later 256/128/96-CU profiles can reuse the
model shard without scattering device assumptions through the kernels.
"""

from sgl_amd_mono.muse_spark.gfx1250.profiles import ACTIVE_PROFILE

HIDDEN = 8192
HEAD_DIM = 64
TOTAL_Q_HEADS = 128
TOTAL_KV_HEADS = 16
EXPERTS = 256

TP = 4
EP = 4
LOCAL_Q_HEADS = TOTAL_Q_HEADS // TP
LOCAL_KV_HEADS = TOTAL_KV_HEADS // TP
LOCAL_EXPERTS = EXPERTS // EP

Q_K = LOCAL_Q_HEADS * HEAD_DIM
KV_K = LOCAL_KV_HEADS * HEAD_DIM
Q_OFF = 0
K_OFF = Q_K
V_OFF = K_OFF + KV_K
G_OFF = V_OFF + KV_K
HEAD_ROWS = G_OFF
QKVG_ROWS = G_OFF + Q_K

TOP_K = 8
MOE_HIDDEN = 4096
INTER = 4096
MXFP4_GROUP = 32
WINDOW = 2047
PAGE = 16
RMS_EPS = 1e-5
POST_EPS = 1e-8
ROUTE_EPS = 1e-15
ROUTER_MULT = 0.17677669529663687
ROUTE_SCALE = 1.0
QK_SCALE = 5.4730064863875
SOFTMAX_SCALE = HEAD_DIM**-0.5
LOG2E = 1.4426950408889634
GATE_TEMPERATURE = 0.3
LAYERS = 62
LAYER_SLOTS = 128

# V1 serves one DFlash2 verify request: 8 rows.
DRAFT_TOKENS = 8
MAX_TOKENS = DRAFT_TOKENS
WIDTHS = (DRAFT_TOKENS,)

# The compiler target is wave32.  Runtime confirmation remains behind the GPU
# approval gate; keeping this explicit prevents inheriting gfx950's wave64 math.
WAVE_SIZE = 32
PROFILE = ACTIVE_PROFILE.name
CU_COUNT = ACTIVE_PROFILE.resident_blocks
RESIDENT_BLOCKS = CU_COUNT
LDS_CAPACITY_BYTES = 320 * 1024

assert TOTAL_Q_HEADS % TP == 0
assert TOTAL_KV_HEADS % TP == 0
assert EXPERTS % EP == 0
assert QKVG_ROWS == 4608
