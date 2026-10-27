#!/usr/bin/env python3
"""
DiGR preprocessing v2: interaction logs + RQ-VAE Semantic IDs -> model-ready arrays
=============================================================================
Copy of `Model Training/preprocess_digr.py` with a richer per-event record.  The
difference is one line of layout:

  v1:  (T_i, S_i_1 .. S_i_L)                  1 + L tokens per event
  v2:  (T_i, A_i, [C_i], S_i_1 .. S_i_L)      1 + A + [C] + L tokens per event

  A_i = action token: 0 exposure / other, 1 click, 2 unknown.  `--add_action`,
        on by default.  v1 used the action only to pick anchors and then threw
        it away, so 89.4% of this dataset's events were indistinguishable.
  C_i = clock token: 0 unknown, else 1 + weekday*24 + hour (UTC).
        `--add_clock`, off by default because it costs one token per event.

The SID block stays last inside every slot, so "the last L tokens of a row are
the last event's codes" still holds and `train_digr.py`'s query extraction needs
no change.  `--n_temporal_bins` now defaults to 128 instead of 64.  The v1
layout is still reachable with `--no-add_action --n_temporal_bins 64`.

Phase 0 of the DiGR pipeline.  This script owns *all* data handling; the training
script (`train_digr.py`) only reads the artefacts written here.

Inputs (defaults match the repo snapshots in `data_snapshot.md`)
----------------------------------------------------------------
`--sequence_path` (parquet), one row per user history, in one of two shapes:

  A) Nested (the shape of `data/seq.parquet`):
       user_id : int64
       seq     : list<struct<item_id, action_type, timestamp>>
     Every event is kept in the history: a click, an exposure or a null
     `action_type` all count as one event.  With `--add_action` (the v2 default)
     the action becomes the event's own token, so exposure, click and unknown
     stay distinguishable in the sequence.  A null / unparseable `action_type`
     is mapped to ACTION_UNKNOWN, so it can never become a target either.

  B) Flat (`history_items` / `history_actions` / `history_timestamps` / `target`):
       user_id           : int64 (optional; without it every row gets UNK)
       history_items     : list<int>          RAW item ids
       history_actions   : list<int>          (optional, aligned with items)
       history_timestamps: list<number>       (optional)
       target            : int64              RAW item id
       split             : str                (optional train/val filter)
     `history_actions` is auto-detected (`history_actions` / `history_action_type`
     / `actions` / `action_types`) and read exactly like the nested
     `action_type`; nulls become UNKNOWN and can never be a target (`--clicks_only`
     drops them from the history as well).
     NOTE: `SID Generation/preprocess_sequence_cli.py` also writes
     `history_items`/`target`, but with 1-BASED CATALOG INDICES and no
     `user_id`; remap those indices to raw item ids before using them here.

`--user_feat_path` (optional parquet, one row per user) -- STATIC USER CONTEXT
-----------------------------------------------------------------------------
  Every column except the user id is tokenised into a per-user feature block
  that opens the row (before the history).  The id column is
  `--user_feat_id_col` (auto-detected: user_id / userid / uid / user / ...); the
  tokenised columns are `--user_feat_cols` (default: every other column, in file
  order).  The raw user id itself is never a token.
  * value tokens are per column: column c owns its own contiguous range, so two
    columns can never share a token id even when their values overlap;
  * a scalar column takes one slot; a list column takes one slot per value
    (order preserved, capped at `--max_feat_len` slots, 0 = the longest list
    actually seen); a missing value (null, empty list) takes that column's UNK
    token;
  * a user absent from the file gets every column's UNK token;
  * a duplicated user id is rejected instead of silently picking a row;
  * the per-split `users.npy` still holds the raw user ids (for evaluation).

Anchors -- how many rows one record produces
---------------------------------------------
  * an ANCHOR is an event whose item becomes the label `y`; only clicks
    (`action_type == --action_click_value`) can be anchors;
  * the history of a row is the up to `--window` (default 20) events directly
    before its anchor, whatever their action type;
  * anchors are taken from the right: a click is accepted only when it lies at
    least `--anchor_gap` (default 5) events before the last accepted anchor, so
    two rows of one record never share almost the same history; a skipped click
    does not consume the quota;
  * at most `--anchors_per_record` (default 3) rows per record: a record with a
    single click still yields one row (provided it has `--min_history` events
    before that click), a record without any click yields none;
  * the val split keeps one anchor per record (its last click) so metrics stay
    cheap and comparable; the infer split has no label and keeps the whole
    windowed history;
  * a record that carries an explicit `target` column yields exactly one row for
    that target, whatever the mode.

Rows are tagged with their history length, so `stats.json` reports how many
records were limited by the window, by `--min_history`, by `--anchor_gap` and by
`--anchors_per_record` (one binding limit per record explains every row loss).

`--catalog_path` (parquet or json) is the item universe and its SID codes:
       item_id : int64 / string
       sids    : list<int> of length L   (or wide `sid_1 .. sid_L`)
  The RQ-VAE CLI dump `index_rqvae_prod.json` (`{"26": [76, 73, 51, 14, 0], ...}`)
  is accepted directly.  Its 5th code is the `--add_unique_id` collision rank
  (the rank of the item inside its collision group), not a semantic level.
  `--num_sid_levels N` keeps only the first N levels; 0 (the default) keeps all
  of them.  A v2 SID such as 2 x 8,192 has no collision rank at all, so pass
  `--num_sid_levels 2`.  With no suffix code, two items that share a code can
  never be ranked apart, so the collision share logged here is exactly the
  fraction of labels whose rank is a coin flip.  Statistics (distinct codes,
  items sharing one, largest group) go to
  `preprocess_meta.json["catalog_collisions"]`, and a share above 1% raises a
  warning.

Timestamps may be epoch seconds / ms / us; the scale is detected from the
median magnitude.  Rows whose target is not in the catalog are dropped, unknown
history items are dropped (and counted).

Outputs (`--output_dir`, default `digr_data/`)
---------------------------------------------
  vocab.json                     token layout (offsets, sizes, total_size)
  catalog_item_ids.npy           catalog order == the label space y
  catalog_sid_codes.npy          [|I|, L] int32 SID codes
  catalog_meta.json              {is_int, size, num_levels}
  user_feat_meta.json            static-feature columns, cardinalities,
                                 block lengths, value counters
  users.npy / users_meta.json    legacy empty profile vocabulary
                                 (read by train_digr.py, holds no user)
  gap_edges.npy                  temporal bin edges
  train/ val/ [infer/]           tokens.npy, offsets.npy, targets.npy,
                                 users.npy, stats.json  (int32 CSR per row)
  preprocess_meta.json           manifest: sizes, max_seq_len (= --window),
                                 window, anchors_per_record, anchor_gap,
                                 sources, counts, catalog_collisions

`stats.json` per split: rows_read / rows_kept / rows_proposed / rows_short /
rows_window_truncated / rows_gap_limited / rows_capped_by_anchors /
rows_with_single_anchor / rows_without_click / rows_missing_target /
rows_non_monotonic / unknown_items / unknown_actions / non_click_actions /
feat_tokens / rows_not_in_feat_file / tokens / avg_items / anchors_per_record.

Token layout written per row (one slot per interaction):
  [F_1 .. F_M, SEP, (T_1, A_1, [C_1], S_1_1 .. S_1_L), ...]
  len = 1 + M + T*slot, slot = 1 + A + C + L  (= vocab.event_width)
  T <= --window: the events of history, newest last, anchor excluded.
  F_j = offset_features + local id of the j-th slot of the user's feature block
        (M slots, one block per user, M = 0 without --user_feat_path)
  T_i = offset_temporal + log-spaced gap bin (first interaction = bin 0)
  A_i = offset_action + {0 exposure, 1 click, 2 unknown}   (0 tokens if off)
  C_i = offset_clock + {0 unknown, 1 + weekday*24 + hour} (0 tokens if off)
  S_i = sid_offsets[l] + code_l
  A 2-level SID with an action token and no clock is slot = 4, against v1's 6
  for a 5-level SID, so the richer event is also the shorter one.

The training script never re-tokenises; it loads these arrays.  This file needs
only numpy + pyarrow (no torch, no transformers, no GPU).

QUICK START
-----------
  python preprocess_digr.py \
      --sequence_path data/seq.parquet \
      --catalog_path data/index_rqvae_prod.json \
      --output_dir digr_data

  # then
  python train_digr.py --mode train --data_dir digr_data --batch_size 32 --compile
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyarrow is required for parquet input/output.") from exc


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SEED = 42
DEFAULT_SEQUENCE_PATH = "data/seq.parquet"
DEFAULT_CATALOG_PATH = "data/index_rqvae_prod.json"
DEFAULT_OUTPUT_DIR = "digr_data"

DEFAULT_WINDOW = 20                 # events of history kept before an anchor
DEFAULT_MIN_HISTORY = 5             # events of history a row must have
DEFAULT_ANCHORS_PER_RECORD = 3      # upper bound on rows per record
DEFAULT_ANCHOR_GAP = 5              # events between two anchors of one record
DEFAULT_N_TEMPORAL_BINS = 128        # v2: 64 in v1.  p50 gap 6,462s, p90 86,280s
DEFAULT_ADD_ACTION = True           # v2: per-event action token
DEFAULT_ADD_CLOCK = False           # v2: hour/weekday, one more token per event
CLOCK_MIN_EPOCH = 946684800         # 2000-01-01; anything earlier is "no clock"
DEFAULT_MAX_GAP_SECONDS = 30 * 24 * 3600
DEFAULT_MAX_FEAT_LEN = 0            # 0 = the longest list actually seen
FEATURE_ID_CANDIDATES = ("user_id", "userid", "uid", "user",
                         "user_key", "visitor_id", "id")
DEFAULT_READ_BATCH_SIZE = 8192
DEFAULT_VAL_RATIO = 0.02
DEFAULT_NUM_SID_LEVELS = 0         # 0 = keep every catalog level, incl. the
#                                    --add_unique_id collision rank

# action_type is normalised into these three codes before any filtering.
ACTION_CLICK = 1
ACTION_OTHER = 0
ACTION_UNKNOWN = -1                # null / unparseable action_type

# v2: how those codes become tokens.  The order is fixed and published in
# vocab.json under "action_token_order", so nothing downstream has to guess.
ACTION_EXPOSURE_TOKEN = 0
ACTION_CLICK_TOKEN = 1
ACTION_UNKNOWN_TOKEN = 2
N_ACTION_TOKENS = 3

# v2: clock token.  0 = unknown timestamp, then 1 + weekday*24 + hour (UTC).
N_CLOCK_TOKENS = 1 + 7 * 24

# What a row's target is.  See the "Anchors" section of the module docstring.
MODE_TRAIN = "train"                # up to --anchors_per_record clicks per record
MODE_VAL = "val"                    # the last click of each record
MODE_INFER = "infer"                # no label, the whole windowed history

BANNER_WIDTH = 88
MANIFEST_NAME = "preprocess_meta.json"
# Files that must exist inside digr_data/<split>/ for `load_split` to accept it.
SPLIT_FILES = ("tokens.npy", "offsets.npy", "targets.npy", "users.npy",
               "stats.json")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_dir: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("DiGR-preprocess")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "preprocess.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def log_banner(logger: logging.Logger, title: str):
    logger.info("=" * BANNER_WIDTH)
    logger.info(title.center(BANNER_WIDTH))
    logger.info("=" * BANNER_WIDTH)


def log_card(logger: logging.Logger, title: str, items: Dict[str, Any]):
    logger.info("+" + "-" * (BANNER_WIDTH - 2) + "+")
    logger.info(f"| {title.upper():<{BANNER_WIDTH - 4}} |")
    logger.info("+" + "-" * (BANNER_WIDTH - 2) + "+")
    for k, v in items.items():
        kk = f"  - {k}:"
        vv = str(v)
        pad = BANNER_WIDTH - len(kk) - len(vv) - 4
        logger.info(f"|{kk} {' ' * max(0, pad)}{vv} |")
    logger.info("+" + "-" * (BANNER_WIDTH - 2) + "+")


def human_int(n: int) -> str:
    return f"{int(n):,}"


# ==== BEGIN MOVED DATA CODE (single-sourced from the old train_digr.py) ====

# ---------------------------------------------------------------------------
# Small numeric / arrow helpers
# ---------------------------------------------------------------------------

_INT64_SENTINEL = np.int64(-9223372036854775808)


def _col_to_numpy(col) -> np.ndarray:
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    if pa.types.is_dictionary(col.type):
        return np.array(col.to_pylist(), dtype=object)
    return col.to_numpy(zero_copy_only=False)


def normalize_ids(values) -> Tuple[np.ndarray, bool]:
    """Normalise an id collection to int64 (preferred) or a str object array.

    Nulls / unparsable values become `_INT64_SENTINEL` instead of demoting the
    whole column to strings, so one bad row cannot change the id dtype (and with
    it the catalog item type and the predicted item ids).
    """
    arr = values if isinstance(values, np.ndarray) else np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)

    if arr.dtype.kind in "iu":
        return arr.astype(np.int64, copy=False), True
    if arr.dtype.kind == "f":
        integral = np.isfinite(arr) & (np.floor(arr) == arr)
        if integral.all():
            return arr.astype(np.int64), True
        out = np.full(len(arr), _INT64_SENTINEL, dtype=np.int64)
        out[integral] = arr[integral].astype(np.int64)
        return out, True

    objs = arr.tolist()
    ints = np.full(len(objs), _INT64_SENTINEL, dtype=np.int64)
    n_parsed = 0
    for i, v in enumerate(objs):
        if v is None:
            continue
        if isinstance(v, float) and not np.isfinite(v):
            continue
        try:
            ints[i] = int(v)
            n_parsed += 1
        except (TypeError, ValueError, OverflowError):
            continue
    if n_parsed * 2 >= len(objs):          # majority numeric -> keep int64 ids
        return ints, True
    return np.asarray(["" if v is None else str(v) for v in objs], dtype=object), False


def _to_float_array(values) -> np.ndarray:
    a = np.asarray(values)
    if a.ndim == 0:
        a = a.reshape(1)
    if a.dtype.kind in "iuf":
        return a.astype(np.float64, copy=False)
    out: List[float] = []
    for v in a.tolist():
        if v is None:
            out.append(0.0)
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(0.0)
    return np.asarray(out, dtype=np.float64)


def _ts_to_seconds(ts: np.ndarray) -> np.ndarray:
    """Detect epoch unit from magnitude and rescale to seconds."""
    if ts.size == 0:
        return ts
    finite = ts[np.isfinite(ts)]
    if finite.size == 0:
        return np.zeros_like(ts)
    mag = float(np.median(np.abs(finite)))
    if mag > 1e17:
        return ts * 1e-9
    if mag > 1e14:
        return ts * 1e-6
    if mag > 1e11:
        return ts * 1e-3
    return ts


def _clock_codes(ts: np.ndarray) -> np.ndarray:
    """Hour-of-day x day-of-week bucket of each event, 0 = unknown timestamp.

    `ts` is epoch seconds (already rescaled by `_ts_to_seconds`), so the bucket
    is UTC.  1970-01-01 was a Thursday, hence the +3 that puts Monday at 0.  A
    timestamp before `CLOCK_MIN_EPOCH` means the field was missing, and gets
    slot 0 so it can never be read as midnight in the first week of 1970.
    """
    n = len(ts)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    out = np.zeros(n, dtype=np.int64)
    valid = np.isfinite(ts) & (ts > CLOCK_MIN_EPOCH)
    if not valid.any():
        return out
    seconds = ts[valid].astype(np.int64)
    days = seconds // 86400
    dow = (days + 3) % 7
    hour = (seconds % 86400) // 3600
    out[valid] = 1 + dow * 24 + hour
    return out


def _lengths_array(col) -> np.ndarray:
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    if isinstance(col, pa.FixedSizeListArray):
        return np.full(len(col), int(col.type.list_size), dtype=np.int64)
    arr = pc.list_value_length(col)
    a = arr.to_numpy(zero_copy_only=False)
    if a.dtype.kind == "f":
        a = np.nan_to_num(a, nan=0.0)
    elif a.dtype.kind == "O":
        a = np.asarray([0 if v is None else int(v) for v in a.tolist()], dtype=np.int64)
    return a.astype(np.int64, copy=False)


def _flatten_list(col):
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    # `list_flatten` also handles FixedSizeListArray and array slices safely.
    return pc.list_flatten(col)


def _lists_from_pylist(col, as_float: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback for exotic list encodings (null lists / null list values)."""
    rows = col.to_pylist()
    lens = np.asarray([len(r) if r else 0 for r in rows], dtype=np.int64)
    flat: List[Any] = []
    for r in rows:
        if r:
            flat.extend(r)
    if as_float:
        return _to_float_array(flat), lens
    return np.asarray(flat), lens


def _safe_int(v) -> np.int64:
    try:
        return np.int64(int(v))
    except (TypeError, ValueError, OverflowError):
        return _INT64_SENTINEL


# ---------------------------------------------------------------------------
# Id index / user map
# ---------------------------------------------------------------------------

class IdIndex:
    """Sorted int64 searchsorted index, or a string dict fallback."""

    def __init__(self, ids, is_int: bool, prepared: bool = False):
        self.is_int = bool(is_int)
        if self.is_int:
            ids = np.asarray(ids, dtype=np.int64)
            if not prepared:
                ids = np.sort(ids)
                keep = np.ones(len(ids), dtype=bool)
                if len(ids) > 1:
                    keep[1:] = ids[1:] != ids[:-1]
                ids = ids[keep]
            self.ids = ids
            self.map = None
        else:
            self.ids = np.asarray([str(v) for v in ids], dtype=object)
            if not prepared:
                self.ids = np.asarray(sorted(set(self.ids.tolist())), dtype=object)
            self.map = {v: i for i, v in enumerate(self.ids.tolist())}

    def __len__(self) -> int:
        return len(self.ids)

    def lookup(self, values) -> np.ndarray:
        vals = values if isinstance(values, np.ndarray) else np.asarray(values)
        if vals.ndim == 0:
            vals = vals.reshape(1)
        n = len(vals)
        if n == 0:
            return np.empty(0, dtype=np.int32)
        if len(self.ids) == 0:
            return np.full(n, -1, dtype=np.int32)

        if self.is_int:
            if vals.dtype.kind == "f":
                q = np.asarray(
                    [_safe_int(v) if np.isfinite(v) else _INT64_SENTINEL
                     for v in vals.tolist()],
                    dtype=np.int64,
                )
            else:
                try:
                    q = vals.astype(np.int64, copy=False)
                except (TypeError, ValueError, OverflowError):
                    q = np.asarray([_safe_int(v) for v in vals.tolist()], dtype=np.int64)
            pos = np.searchsorted(self.ids, q)
            pos = np.clip(pos, 0, len(self.ids) - 1)
            ok = self.ids[pos] == q
            out = pos.astype(np.int32)
            out[~ok] = -1
            return out
        return np.fromiter(
            (self.map.get(str(v), -1) for v in vals.tolist()),
            dtype=np.int32,
            count=n,
        )


class UserMap:
    """Legacy dense profile index (kept so old train checkpoints still load).

    Users are no longer tokenised: static user context comes from
    `--user_feat_path`.  New datasets write an empty map here.
    """

    def __init__(self, users, is_int: bool):
        self.index = IdIndex(users, is_int, prepared=False)
        self.is_int = self.index.is_int
        self.n_profile = len(self.index) + 1

    def lookup(self, values) -> np.ndarray:
        return (self.index.lookup(values) + 1).astype(np.int32)

    def save(self, out_dir: str):
        np.save(os.path.join(out_dir, "users.npy"), self.index.ids)
        with open(os.path.join(out_dir, "users_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"is_int": bool(self.is_int), "n_profile": int(self.n_profile)}, f)

    @classmethod
    def load(cls, out_dir: str) -> "UserMap":
        with open(os.path.join(out_dir, "users_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        users = np.load(os.path.join(out_dir, "users.npy"), allow_pickle=True)
        return cls(users, bool(meta["is_int"]))


@dataclass
class FeatureColumn:
    """One tokenised `--user_feat_path` column.

    `cardinality` distinct values occupy `cardinality` tokens; the next id is
    that column's UNK slot, used for a missing value (null, empty list or a code
    that never appeared).  Columns never share ids, so two columns with the same
    values stay distinguishable.
    """

    name: str
    is_list: bool
    cardinality: int
    max_len: int = 1            # slots one row may take (list cap; 1 for scalars)

    @property
    def size(self) -> int:
        return int(self.cardinality) + 1     # values + one UNK slot


class UserFeatureMap:
    """Static user attributes as one flat, per-user token block (CSR).

    Row `u` owns `codes[offsets[u]:offsets[u + 1]]`: for every column, in file
    order, its values (list) or its single value (scalar), already mapped to a
    LOCAL token id in `[0, n_local)`.  The final token id is
    `vocab.offset_features + local id`, which is why the local layout (values of
    a column, then its UNK slot, columns back to back) must match the vocabulary
    layout exactly -- `Vocab.feature_sizes` is built from `columns`.

    The last row (`n_rows`) is synthetic and holds every column's UNK slot, so a
    user that is absent from the file still gets a full block: the block length
    stays a property of the columns, and the row offsets stay a plain cumsum.
    """

    def __init__(self, columns, user_ids, is_int: bool, codes, offsets,
                 counters: Optional[Dict[str, Any]] = None,
                 path: Optional[str] = None, id_column: Optional[str] = None):
        self.columns: List[FeatureColumn] = list(columns)
        self.codes = np.asarray(codes, dtype=np.int32)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.index = IdIndex(user_ids, is_int, prepared=False)
        self.counters: Dict[str, int] = dict(counters or {})
        self.path = path
        self.id_column = id_column

        self.local_bases: List[int] = []
        cursor = 0
        for col in self.columns:
            self.local_bases.append(cursor)
            cursor += col.size
        self.n_local = cursor
        if len(self.offsets) != self.n_rows + 2:
            raise ValueError(
                f"feature offsets must describe {self.n_rows} users + 1 UNK row, "
                f"got {len(self.offsets) - 1} rows"
            )

    @classmethod
    def empty(cls) -> "UserFeatureMap":
        return cls([], np.empty(0, dtype=np.int64), True,
                   np.zeros(0, dtype=np.int32), np.zeros(2, dtype=np.int64))

    @property
    def n_rows(self) -> int:
        return len(self.offsets) - 2          # the last row is the UNK block

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    @property
    def max_len(self) -> int:
        if len(self.offsets) < 2:
            return 0
        return int(np.max(np.diff(self.offsets))) if len(self.offsets) > 1 else 0

    def lookup(self, raw_users) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Raw user ids -> (row index, block start, block length)."""
        idx = self.index.lookup(raw_users)
        idx = np.where(idx < 0, self.n_rows, idx).astype(np.int64)
        start = self.offsets[idx]
        return idx, start, self.offsets[idx + 1] - start

    def meta(self) -> Dict[str, Any]:
        lens = np.diff(self.offsets) if len(self.offsets) > 1 else np.zeros(1, np.int64)
        return {
            "path": self.path,
            "id_column": self.id_column,
            "n_users": int(self.n_rows),
            "n_columns": int(self.n_columns),
            "columns": [
                {"name": c.name, "is_list": bool(c.is_list),
                 "cardinality": int(c.cardinality), "max_len": int(c.max_len),
                 "size": int(c.size), "local_base": int(b)}
                for c, b in zip(self.columns, self.local_bases)
            ],
            "block_len_min": int(lens.min()),
            "block_len_max": int(lens.max()),
            "block_len_mean": round(float(lens.mean()), 2),
            "counters": dict(self.counters),
        }

    def save(self, out_dir: str) -> None:
        with open(os.path.join(out_dir, "user_feat_meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump(self.meta(), f, indent=2)


def _feature_values(col, name: str, path: str) -> Tuple[np.ndarray, np.ndarray]:
    """(int64 values in row order, slots per row) of one feature column.

    Nulls become `_INT64_SENTINEL`; they keep their slot and turn into the
    column's UNK token later.  A scalar column always takes exactly one slot, so
    its block length does not depend on whether the value is present.
    """
    if _is_list_type(col.type):
        lens = _lengths_array(col)
        vals, is_int = normalize_ids(_col_to_numpy(_flatten_list(col)))
        if not is_int:
            raise SystemExit(
                f"{path}: column '{name}' holds non-numeric values; integers only "
                "(drop it with --user_feat_cols)"
            )
        if len(vals) != int(lens.sum()):
            raise ValueError(
                f"{path}: column '{name}' flattening mismatch "
                f"({len(vals)} values for {int(lens.sum())} slots)"
            )
        return vals, lens
    vals, is_int = normalize_ids(_col_to_numpy(col))
    if not is_int:
        raise SystemExit(
            f"{path}: column '{name}' holds non-numeric values; integers only "
            "(drop it with --user_feat_cols)"
        )
    return vals, np.ones(len(vals), dtype=np.int64)


def _encode_feature_column(raw, lens, col: FeatureColumn, values: np.ndarray,
                           local_base: int, counters: Dict[str, int]
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """-> (slots per row, local token per slot) for one column of one batch."""
    if col.is_list:
        take = np.minimum(lens, int(col.max_len))
    else:
        take = np.ones_like(lens)
    counters["values_truncated"] += int((lens - take).sum())

    absent = take <= 0
    slot_len = np.where(absent, 1, take).astype(np.int64)
    row_base = np.concatenate([[0], np.cumsum(slot_len)[:-1]]).astype(np.int64)
    out = np.empty(int(slot_len.sum()), dtype=np.int32)
    unk_tok = np.int32(local_base + int(col.cardinality))
    if absent.any():
        out[row_base[absent]] = unk_tok

    keep_n = np.where(absent, 0, take).astype(np.int64)
    n_keep = int(keep_n.sum())
    if n_keep:
        row_of = np.repeat(np.arange(len(lens), dtype=np.int64), keep_n)
        kbase = np.concatenate([[0], np.cumsum(keep_n)[:-1]]).astype(np.int64)
        within = np.arange(n_keep, dtype=np.int64) - kbase[row_of]
        src = np.concatenate([[0], np.cumsum(lens)[:-1]])[row_of] + within
        dst = row_base[row_of] + within
        vals = raw[src]
        if len(values):
            pos = np.clip(np.searchsorted(values, vals), 0, len(values) - 1)
            known = values[pos] == vals
        else:
            pos = np.zeros(len(vals), dtype=np.int64)
            known = np.zeros(len(vals), dtype=bool)
        counters["values_unknown"] += int(
            (~known & (vals != _INT64_SENTINEL)).sum()
        )
        out[dst] = np.where(known, local_base + pos, unk_tok).astype(np.int32)
    return slot_len, out


def _interleave_feature_blocks(slot_lens: List[np.ndarray],
                               col_tokens: List[np.ndarray]
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Column blocks (all rows of one column) -> per-row blocks (all columns)."""
    n = len(slot_lens[0]) if slot_lens else 0
    rows = np.arange(n, dtype=np.int64)
    row_tot = np.zeros(n, dtype=np.int64)
    for sl in slot_lens:
        row_tot += sl
    row_base = np.concatenate([[0], np.cumsum(row_tot)[:-1]]).astype(np.int64)
    block = np.empty(int(row_tot.sum()), dtype=np.int32)
    pre = np.zeros(n, dtype=np.int64)          # slots taken by earlier columns
    for sl, toks in zip(slot_lens, col_tokens):
        n_slots = int(toks.size)
        if n_slots:
            row_of = np.repeat(rows, sl)
            cbase = np.concatenate([[0], np.cumsum(sl)[:-1]]).astype(np.int64)
            within = np.arange(n_slots, dtype=np.int64) - cbase[row_of]
            block[row_base[row_of] + pre[row_of] + within] = toks
        pre = pre + sl
    return block, row_tot


def _detect_feature_id_col(names: Sequence[str], explicit: Optional[str]) -> str:
    if explicit is not None:
        if explicit not in names:
            raise SystemExit(
                f"--user_feat_id_col '{explicit}' is not a column of the feature "
                f"file ({list(names)})"
            )
        return explicit
    for cand in FEATURE_ID_CANDIDATES:
        if cand in names:
            return cand
    for name in names:
        if "user" in name.lower():
            return name
    raise SystemExit(
        f"cannot find the user id column among {list(names)}; "
        "pass --user_feat_id_col"
    )


def build_user_feature_map(path: str, args, logger: logging.Logger) -> UserFeatureMap:
    """Read `--user_feat_path` into a per-user token block (two passes)."""
    t0 = time.time()
    pf = pq.ParquetFile(path)
    names = list(pf.schema_arrow.names)
    id_col = _detect_feature_id_col(names, getattr(args, "user_feat_id_col", None))
    wanted = ([str(c) for c in args.user_feat_cols]
              if getattr(args, "user_feat_cols", None)
              else [n for n in names if n != id_col])
    if id_col in wanted:
        raise SystemExit(f"--user_feat_cols must not contain the id column '{id_col}'")
    for name in wanted:
        if name not in names:
            raise SystemExit(
                f"--user_feat_cols: '{name}' is not a column of {path} ({names})"
            )
    if not wanted:
        raise SystemExit(f"{path} has no column to tokenise besides '{id_col}'")

    cap = int(getattr(args, "max_feat_len", 0) or 0)
    read = int(getattr(args, "read_batch_size", DEFAULT_READ_BATCH_SIZE))

    # --- pass 1: every value that exists becomes a token of its column -------
    seen: Dict[str, set] = {name: set() for name in wanted}
    is_list: Dict[str, bool] = {name: False for name in wanted}
    obs_max: Dict[str, int] = {name: 0 for name in wanted}
    counters: Dict[str, int] = {"values_null": 0, "values_unknown": 0,
                                "values_truncated": 0}
    rows_seen = 0
    for batch in pf.iter_batches(batch_size=read, columns=wanted):
        rows_seen += batch.num_rows
        for name in wanted:
            col = batch.column(batch.schema.get_field_index(name))
            is_list[name] = _is_list_type(col.type)
            vals, lens = _feature_values(col, name, path)
            bad = vals == _INT64_SENTINEL
            counters["values_null"] += int(bad.sum())
            good = vals[~bad]
            if good.size:
                seen[name].update(np.unique(good).tolist())
            if lens.size:
                obs_max[name] = max(obs_max[name], int(lens.max()))

    columns: List[FeatureColumn] = []
    values: List[np.ndarray] = []
    local_bases: List[int] = []
    cursor = 0
    for name in wanted:
        vals = np.asarray(sorted(seen[name]), dtype=np.int64)
        if is_list[name]:
            length = cap if cap > 0 else max(1, obs_max[name])
        else:
            length = 1
        columns.append(FeatureColumn(name=name, is_list=bool(is_list[name]),
                                     cardinality=int(len(vals)),
                                     max_len=int(length)))
        values.append(vals)
        local_bases.append(cursor)
        cursor += len(vals) + 1

    # --- pass 2: encode every user into its block ---------------------------
    code_parts: List[np.ndarray] = []
    len_parts: List[np.ndarray] = []
    user_parts: List[np.ndarray] = []
    null_id_rows = 0
    for batch in pf.iter_batches(batch_size=read, columns=[id_col] + wanted):
        users, id_is_int = normalize_ids(_col_to_numpy(batch.column(0)))
        if not id_is_int:
            raise SystemExit(
                f"{path}: '{id_col}' holds non-numeric ids; integers only "
                "(pass --user_feat_id_col if this is not the user id)"
            )
        keep_rows = users != _INT64_SENTINEL
        if not keep_rows.all():
            null_id_rows += int((~keep_rows).sum())
            users = users[keep_rows]
            batch = batch.filter(pa.array(keep_rows))
        if batch.num_rows == 0:
            continue
        slot_lens: List[np.ndarray] = []
        col_tokens: List[np.ndarray] = []
        for i, (col_def, vals, base) in enumerate(zip(columns, values, local_bases)):
            raw, lens = _feature_values(batch.column(i + 1), col_def.name, path)
            sl, toks = _encode_feature_column(raw, lens, col_def, vals, base, counters)
            slot_lens.append(sl)
            col_tokens.append(toks)
        block_codes, row_tot = _interleave_feature_blocks(slot_lens, col_tokens)
        code_parts.append(block_codes)
        len_parts.append(row_tot)
        user_parts.append(users)

    codes = np.concatenate(code_parts) if code_parts else np.zeros(0, np.int32)
    row_tot = np.concatenate(len_parts) if len_parts else np.zeros(0, np.int64)
    users = np.concatenate(user_parts) if user_parts else np.zeros(0, np.int64)
    if len(users):
        uniq, counts = np.unique(users, return_counts=True)
        dup = uniq[counts > 1]
        if dup.size:
            raise SystemExit(
                f"{path}: user id {int(dup[0])} appears {int(counts[counts > 1][0])} "
                f"times ({int(dup.size)} duplicated ids); the feature file needs "
                "exactly one row per user"
            )

    offsets = np.concatenate([[0], np.cumsum(row_tot)]).astype(np.int64)
    absent = np.asarray([base + col.cardinality
                         for base, col in zip(local_bases, columns)], dtype=np.int32)
    codes = np.concatenate([codes.astype(np.int32), absent])
    offsets = np.concatenate([offsets, np.asarray([offsets[-1] + len(absent)],
                                                  dtype=np.int64)])

    counters["rows_read"] = int(rows_seen)
    counters["rows_null_id"] = int(null_id_rows)
    feat_map = UserFeatureMap(columns, users, True, codes, offsets, counters,
                              path=path, id_column=id_col)
    logger.info(
        f"  + user features from {path}: {human_int(feat_map.n_rows)} users x "
        f"{len(columns)} columns -> {human_int(cursor)} tokens "
        f"(block {int(np.diff(offsets).min())}-{feat_map.max_len}, "
        f"{int(codes.size)} codes, {time.time() - t0:.1f}s)"
    )
    for col in columns:
        logger.info(
            f"      {col.name}: {'list' if col.is_list else 'scalar'}, "
            f"{human_int(col.cardinality)} values (+UNK), "
            f"max {human_int(col.max_len)} slots/row"
        )
    if counters["values_truncated"]:
        logger.info(
            f"      ! {human_int(counters['values_truncated'])} values truncated "
            f"by --max_feat_len {cap}"
        )
    if null_id_rows:
        logger.info(f"      ! {human_int(null_id_rows)} rows without a user id skipped")
    return feat_map


# ---------------------------------------------------------------------------
# Phase 1: vocabulary partition + catalog
# ---------------------------------------------------------------------------

@dataclass
class Vocab:
    """Contiguous, disjoint token ranges (the table from the plan).

    `feature_sizes[c]` is the size of static-feature column c: its distinct
    values plus one UNK slot at the end.  Columns keep separate ranges, so two
    columns never share a token id even when their values overlap.
    """

    sid_sizes: List[int]
    n_temporal: int
    feature_sizes: Optional[List[int]] = None
    n_action: int = 0           # v2: per-event action token, 0 = not emitted
    n_clock: int = 0            # v2: per-event hour/weekday token, 0 = not emitted
    n_special: int = 2          # PAD = 0, SEP/BOS = 1
    pad_id: int = 0
    sep_id: int = 1

    def __post_init__(self):
        if self.n_special < 2:
            raise ValueError("n_special must be >= 2 (PAD + SEP)")
        if self.n_temporal < 1:
            raise ValueError("n_temporal must be >= 1")
        self.sid_sizes = [int(s) for s in self.sid_sizes]
        if any(s < 1 for s in self.sid_sizes):
            raise ValueError(f"invalid SID level sizes: {self.sid_sizes}")
        self.n_action = max(0, int(self.n_action))
        self.n_clock = max(0, int(self.n_clock))
        self.feature_sizes = [int(s) for s in (self.feature_sizes or [])]
        if any(s < 2 for s in self.feature_sizes):
            raise ValueError(
                f"invalid feature sizes (a column needs 1 value + 1 UNK): "
                f"{self.feature_sizes}"
            )

        cursor = int(self.n_special)
        self.sid_offsets: List[int] = []
        for size in self.sid_sizes:
            self.sid_offsets.append(cursor)
            cursor += size
        self.offset_temporal = cursor
        cursor += int(self.n_temporal)
        self.offset_action = cursor
        cursor += int(self.n_action)
        self.offset_clock = cursor
        cursor += int(self.n_clock)
        self.offset_features = cursor
        self.feature_offsets: List[int] = []
        for size in self.feature_sizes:
            self.feature_offsets.append(cursor)
            cursor += size
        self.total_size = cursor

    @property
    def num_levels(self) -> int:
        return len(self.sid_sizes)

    @property
    def num_features(self) -> int:
        return len(self.feature_sizes)

    @property
    def event_width(self) -> int:
        """Tokens one event occupies: gap, [action], [clock], then its codes."""
        return (1 + (1 if self.n_action else 0) + (1 if self.n_clock else 0)
                + self.num_levels)

    @property
    def feature_unk_ids(self) -> List[int]:
        return [off + size - 1
                for off, size in zip(self.feature_offsets, self.feature_sizes)]

    def feature_local_bases(self) -> List[int]:
        """Where each column starts inside the feature block (row relative)."""
        return [off - self.offset_features for off in self.feature_offsets]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sid_sizes": list(self.sid_sizes),
            "n_temporal": int(self.n_temporal),
            "n_action": int(self.n_action),
            "n_clock": int(self.n_clock),
            "offset_action": int(self.offset_action),
            "offset_clock": int(self.offset_clock),
            "event_width": int(self.event_width),
            "action_token_order": ["exposure", "click", "unknown"],
            "feature_sizes": list(self.feature_sizes),
            "n_special": int(self.n_special),
            "pad_id": int(self.pad_id),
            "sep_id": int(self.sep_id),
            "sid_offsets": list(self.sid_offsets),
            "offset_temporal": int(self.offset_temporal),
            "offset_features": int(self.offset_features),
            "feature_offsets": list(self.feature_offsets),
            "feature_unk_ids": list(self.feature_unk_ids),
            "total_size": int(self.total_size),
            "num_levels": int(self.num_levels),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Vocab":
        return cls(
            sid_sizes=list(d["sid_sizes"]),
            n_temporal=int(d["n_temporal"]),
            n_action=int(d.get("n_action", 0)),
            n_clock=int(d.get("n_clock", 0)),
            feature_sizes=list(d.get("feature_sizes") or []),
            n_special=int(d.get("n_special", 2)),
            pad_id=int(d.get("pad_id", 0)),
            sep_id=int(d.get("sep_id", 1)),
        )


class ItemCatalog:
    """Static item universe + SID codes. Order defines the label space y."""

    def __init__(self, item_ids: np.ndarray, is_int: bool, sid_codes: np.ndarray,
                 prepared: bool = False):
        self.item_ids = np.asarray(item_ids)
        self.is_int = bool(is_int)
        self.sid_codes = np.asarray(sid_codes, dtype=np.int32)
        if self.sid_codes.ndim != 2:
            raise ValueError("sid_codes must have shape [n_items, num_levels]")
        if len(self.item_ids) != len(self.sid_codes):
            raise ValueError("item_ids and sid_codes length mismatch")

        if prepared:
            self.index = IdIndex(self.item_ids, self.is_int, prepared=True)
        else:
            self.item_ids, self.sid_codes = self._sort_and_dedupe()
            self.index = IdIndex(self.item_ids, self.is_int, prepared=True)

    def _sort_and_dedupe(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_int:
            ids = np.asarray(self.item_ids, dtype=np.int64)
            order = np.argsort(ids, kind="stable")
            ids = ids[order]
            sids = self.sid_codes[order]
            keep = np.ones(len(ids), dtype=bool)
            if len(ids) > 1:
                keep[1:] = ids[1:] != ids[:-1]
            return ids[keep], sids[keep]

        seen: Dict[str, int] = {}
        out_ids: List[Any] = []
        out_rows: List[int] = []
        for i, v in enumerate(self.item_ids.tolist()):
            s = str(v)
            if s in seen:
                continue
            seen[s] = len(out_ids)
            out_ids.append(v)
            out_rows.append(i)
        ids = np.asarray(out_ids, dtype=object) if out_ids else np.empty(0, dtype=object)
        sids = self.sid_codes[out_rows] if out_rows else np.zeros(
            (0, self.sid_codes.shape[1]), dtype=np.int32
        )
        return ids, sids

    @property
    def size(self) -> int:
        return len(self.item_ids)

    @property
    def num_levels(self) -> int:
        return int(self.sid_codes.shape[1])

    def lookup(self, values) -> np.ndarray:
        return self.index.lookup(values)

    def save(self, out_dir: str):
        np.save(os.path.join(out_dir, "catalog_item_ids.npy"), self.item_ids)
        np.save(os.path.join(out_dir, "catalog_sid_codes.npy"), self.sid_codes)
        with open(os.path.join(out_dir, "catalog_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "is_int": bool(self.is_int),
                    "size": int(self.size),
                    "num_levels": int(self.num_levels),
                },
                f,
            )

    @classmethod
    def load(cls, out_dir: str) -> "ItemCatalog":
        with open(os.path.join(out_dir, "catalog_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        ids = np.load(os.path.join(out_dir, "catalog_item_ids.npy"), allow_pickle=True)
        sids = np.load(os.path.join(out_dir, "catalog_sid_codes.npy"), allow_pickle=True)
        return cls(ids, bool(meta["is_int"]), sids, prepared=True)


# --- catalog parquet schema detection -------------------------------------

_SID_NUMBERED_RE = re.compile(
    r"^(?:sid|code|semantic_?id|semantic_?code|token_?id|code_?id)[_\-]?(\d+)$",
    re.IGNORECASE,
)
_SID_LIST_NAMES = (
    "sids", "sid", "codes", "semantic_ids", "semantic_id",
    "token_ids", "sid_codes", "tokens",
)
_CATALOG_ITEM_NAMES = ("item_id", "item", "itemid", "id", "rid")


def _find_col(names: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {n.lower(): n for n in names}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def _is_list_type(t) -> bool:
    return (
        pa.types.is_list(t)
        or pa.types.is_large_list(t)
        or pa.types.is_fixed_size_list(t)
    )


def _read_sid_matrix(table: pa.Table,
                     sid_cols: Optional[Sequence[str]],
                     sid_list_col: Optional[str]) -> np.ndarray:
    if sid_list_col is not None:
        col = table.column(sid_list_col)
        if isinstance(col, pa.ChunkedArray):
            col = col.combine_chunks()
        lens = _lengths_array(col)
        if lens.size == 0:
            raise ValueError(f"empty SID list column `{sid_list_col}`")
        if not np.all(lens == lens[0]):
            raise ValueError(
                f"SID list column `{sid_list_col}` is ragged (levels must be fixed)"
            )
        flat = np.asarray(_flatten_list(col))
        mat = flat.reshape(len(col), int(lens[0]))
    else:
        assert sid_cols is not None
        levels = []
        for c in sid_cols:
            a = _col_to_numpy(table.column(c))
            if a.dtype.kind == "f":
                a = np.nan_to_num(a, nan=-1.0)
            levels.append(a.astype(np.int64, copy=False))
        mat = np.stack(levels, axis=1)

    mat = mat.astype(np.int64, copy=False)
    if mat.ndim != 2 or mat.shape[1] < 1:
        raise ValueError(f"could not build a 2-D SID matrix, got shape {mat.shape}")
    if np.any(mat < 0):
        raise ValueError("SID matrix contains negative codes")
    return mat


def load_catalog_from_parquet(path: str, args, logger: logging.Logger) -> ItemCatalog:
    logger.info(f"  > reading catalog: {path}")
    t0 = time.time()
    table = pq.read_table(path)
    names = table.column_names

    item_col = args.catalog_item_col or _find_col(names, _CATALOG_ITEM_NAMES)
    if item_col is None:
        raise ValueError(
            f"could not find an item-id column in {path}; columns={names}. "
            "Pass --catalog_item_col."
        )

    sid_cols: Optional[List[str]] = list(args.catalog_sid_cols) if args.catalog_sid_cols else None
    sid_list_col: Optional[str] = None
    if sid_cols is None:
        lower = {n.lower(): n for n in names}
        for cand in _SID_LIST_NAMES:
            col_name = lower.get(cand)
            if col_name is not None and _is_list_type(table.schema.field(col_name).type):
                sid_list_col = col_name
                break
        if sid_list_col is None:
            numbered = []
            for n in names:
                m = _SID_NUMBERED_RE.match(n) or _SID_NUMBERED_RE.match(n.replace("-", "_"))
                if m:
                    numbered.append((int(m.group(1)), n))
            if not numbered:
                raise ValueError(
                    f"could not detect SID columns in {path}; columns={names}. "
                    "Pass --catalog_sid_cols sid_1 sid_2 ... or a list column."
                )
            numbered.sort(key=lambda x: x[0])
            sid_cols = [n for _, n in numbered]

    sid_mat = _read_sid_matrix(table, sid_cols, sid_list_col)
    item_raw = _col_to_numpy(table.column(item_col))
    item_ids, is_int = normalize_ids(item_raw)

    if len(item_ids) != len(sid_mat):
        raise ValueError(
            f"catalog length mismatch: {len(item_ids)} item ids vs {len(sid_mat)} SID rows"
        )

    catalog = ItemCatalog(item_ids, is_int, sid_mat, prepared=False)
    logger.info(
        f"  + catalog ready: {human_int(catalog.size)} items x "
        f"{catalog.num_levels} SID levels "
        f"(id type={'int64' if catalog.is_int else 'str'}) "
        f"in {time.time() - t0:.1f}s"
    )
    return catalog


def load_catalog_from_json(path: str, args, logger: logging.Logger) -> ItemCatalog:
    """Accept the RQ-VAE artefact directly: {item_id: [c_1, ..., c_L], ...}."""
    logger.info(f"  > reading SID json: {path}")
    t0 = time.time()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    offset = int(args.code_offset)
    if isinstance(data, dict):
        item_raw = list(data.keys())
        code_rows = list(data.values())
    elif isinstance(data, list):
        item_key = args.catalog_item_col or "item_id"
        item_raw = []
        code_rows = []
        for row in data:
            if not isinstance(row, dict):
                raise ValueError(f"{path}: expected JSON objects, got {type(row).__name__}")
            item_raw.append(row.get(item_key))
            code_rows.append(row.get("sids") or row.get("codes"))
    else:
        raise ValueError(f"{path}: unsupported JSON root ({type(data).__name__})")

    try:
        codes = np.asarray(
            [[int(c) - offset for c in row] for row in code_rows], dtype=np.int64
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: every SID entry must be a list of ints") from exc
    if codes.ndim != 2 or codes.shape[1] < 1:
        raise ValueError(f"{path}: could not build a 2-D SID matrix, got {codes.shape}")
    if np.any(codes < 0):
        raise ValueError(f"{path}: negative SID codes after --code_offset {offset}")

    item_ids, is_int = normalize_ids(np.asarray(item_raw, dtype=object))
    if len(item_ids) != len(codes):
        raise ValueError(
            f"catalog length mismatch: {len(item_ids)} item ids vs {len(codes)} SID rows"
        )
    catalog = ItemCatalog(item_ids, is_int, codes, prepared=False)
    logger.info(
        f"  + catalog ready: {human_int(catalog.size)} items x "
        f"{catalog.num_levels} SID levels "
        f"(id type={'int64' if catalog.is_int else 'str'}) "
        f"in {time.time() - t0:.1f}s"
    )
    return catalog


def build_vocab(catalog: ItemCatalog, feat_map: UserFeatureMap, args) -> Vocab:
    if catalog.size == 0:
        raise ValueError("empty catalog")
    maxima = catalog.sid_codes.max(axis=0).astype(np.int64) + 1
    if args.codebook_size is not None:
        sizes = [max(int(args.codebook_size), int(m)) for m in maxima]
    else:
        sizes = [int(m) for m in maxima]
    return Vocab(
        sid_sizes=sizes,
        n_temporal=int(args.n_temporal_bins),
        feature_sizes=[col.size for col in feat_map.columns],
        n_action=N_ACTION_TOKENS if bool(getattr(args, "add_action", False)) else 0,
        n_clock=N_CLOCK_TOKENS if bool(getattr(args, "add_clock", False)) else 0,
    )


def build_gap_edges(n_bins: int, max_gap_seconds: float) -> np.ndarray:
    """Log-spaced gap buckets: 0 = first item / <1s, then 1..n_bins-1."""
    if n_bins <= 1:
        return np.empty(0, dtype=np.float64)
    return np.geomspace(1.0, max(1.0, max_gap_seconds), n_bins - 1, dtype=np.float64)


# ---------------------------------------------------------------------------
# Sequence schema detection
# ---------------------------------------------------------------------------

@dataclass
class SequenceSchema:
    user_col: Optional[str] = None
    nested_col: Optional[str] = None
    history_col: Optional[str] = None
    history_ts_col: Optional[str] = None
    history_actions_col: Optional[str] = None
    target_col: Optional[str] = None
    split_col: Optional[str] = None
    item_key: str = "item_id"
    ts_key: str = "timestamp"
    action_key: str = "action_type"


_USER_NAMES = ("user_id", "user", "uid", "user_idx", "userid")
_HISTORY_NAMES = (
    "history_items", "item_ids", "history_item_ids", "hist_items",
    "items", "item_seq", "history",
)
_TS_NAMES = (
    "history_timestamps", "history_times", "timestamps", "times",
    "history_ts", "ts",
)
_ACTION_LIST_NAMES = (
    "history_actions", "history_action_types", "history_action_type",
    "actions", "action_types",
)
_TARGET_NAMES = (
    "target", "target_item_id", "label", "next_item_id", "y", "target_id",
)
_NESTED_NAMES = ("seq", "sequence", "interactions", "history", "sessions", "events")
_SPLIT_NAMES = ("split", "data_split", "subset", "set")

_ITEM_KEYS = ("item_id", "item", "itemid", "iid", "rid", "id")
_TS_KEYS = ("timestamp", "time", "ts", "log_time", "event_time", "action_time", "click_time")
_ACTION_KEYS = ("action_type", "action", "act", "behavior_type", "event_type")


def _none_if_auto(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if value.strip().lower() in ("", "auto", "none", "null"):
        return None
    return value


def detect_sequence_schema(path: str, args) -> SequenceSchema:
    pf = pq.ParquetFile(path)
    fields = pf.schema_arrow
    names = [f.name for f in fields]
    type_by_name = {f.name: f.type for f in fields}

    explicit_nested = _none_if_auto(args.seq_col)
    explicit_hist = _none_if_auto(args.history_col)

    nested_col = explicit_nested
    if nested_col is None and explicit_hist is None:
        for cand in _NESTED_NAMES:
            col = _find_col(names, (cand,))
            if col is not None:
                t = type_by_name[col]
                if _is_list_type(t) and pa.types.is_struct(t.value_type):
                    nested_col = col
                    break

    history_col = explicit_hist
    if nested_col is None and history_col is None:
        for cand in _HISTORY_NAMES:
            col = _find_col(names, (cand,))
            if col is not None:
                t = type_by_name[col]
                if _is_list_type(t) and not pa.types.is_struct(t.value_type):
                    history_col = col
                    break
    if nested_col is None and history_col is None:
        raise ValueError(
            f"could not detect a sequence column in {path}; columns={names}. "
            "Pass --seq_col (list<struct>) or --history_col (list<int>)."
        )

    if nested_col is not None:
        nested_struct = type_by_name[nested_col].value_type
        struct_fields = [nested_struct.field(i).name for i in range(nested_struct.num_fields)]
        item_key = _none_if_auto(args.item_key) or _find_col(struct_fields, _ITEM_KEYS) or "item_id"
        ts_key = _none_if_auto(args.ts_key) or _find_col(struct_fields, _TS_KEYS) or "timestamp"
        action_key = _none_if_auto(args.action_key) or _find_col(struct_fields, _ACTION_KEYS) or "action_type"
    else:
        item_key = _none_if_auto(args.item_key) or "item_id"
        ts_key = _none_if_auto(args.ts_key) or "timestamp"
        action_key = _none_if_auto(args.action_key) or "action_type"

    user_col = _none_if_auto(args.user_col) or _find_col(names, _USER_NAMES)
    target_col = _none_if_auto(args.target_col) or _find_col(names, _TARGET_NAMES)
    split_col = _none_if_auto(args.split_col) or _find_col(names, _SPLIT_NAMES)

    hist_ts_col = None
    hist_act_col = None
    if nested_col is None:
        hist_ts_col = _none_if_auto(args.history_ts_col) or _find_col(names, _TS_NAMES)
        if hist_ts_col is not None and not _is_list_type(type_by_name[hist_ts_col]):
            hist_ts_col = None
        hist_act_col = _none_if_auto(getattr(args, "history_actions_col", None)) \
            or _find_col(names, _ACTION_LIST_NAMES)
        if hist_act_col is not None and not _is_list_type(type_by_name[hist_act_col]):
            hist_act_col = None

    return SequenceSchema(
        user_col=user_col,
        nested_col=nested_col,
        history_col=history_col,
        history_ts_col=hist_ts_col,
        history_actions_col=hist_act_col,
        target_col=target_col,
        split_col=split_col,
        item_key=item_key,
        ts_key=ts_key,
        action_key=action_key,
    )


def describe_schema(schema: SequenceSchema) -> str:
    return (
        f"nested={schema.nested_col} hist={schema.history_col} "
        f"ts={schema.history_ts_col} acts={schema.history_actions_col} "
        f"target={schema.target_col} user={schema.user_col} split={schema.split_col}"
    )


# ---------------------------------------------------------------------------
# Raw parquet -> ragged arrays
# ---------------------------------------------------------------------------

def _split_matches(value, wanted: Optional[str]) -> bool:
    if wanted is None:
        return True
    if value is None:
        return False
    return str(value).lower().startswith(wanted)


def _get_key(d: Dict[str, Any], keys: Sequence[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _action_code(value: Any, click_value: float) -> int:
    """Normalise one raw `action_type` into ACTION_CLICK / OTHER / UNKNOWN.

    `null` is checked explicitly (None, NaN, empty string, unparseable) instead
    of being compared against the click value, which would silently turn every
    null into "not a click" without anyone being able to count it.
    """
    if value is None:
        return ACTION_UNKNOWN
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "none", "null", "nan", "<na>"):
            return ACTION_UNKNOWN
        try:
            value = float(text)
        except ValueError:
            return ACTION_UNKNOWN
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ACTION_UNKNOWN
    if not np.isfinite(number):
        return ACTION_UNKNOWN
    return ACTION_CLICK if number == click_value else ACTION_OTHER


def _row_slice(keep_rows: np.ndarray, start: int, n: int) -> np.ndarray:
    """Boolean mask for rows [start, start+n); missing rows count as dropped."""
    part = np.asarray(keep_rows[start:start + n], dtype=bool)
    if len(part) == n:
        return part
    padded = np.zeros(n, dtype=bool)
    padded[: len(part)] = part
    return padded


def _val_row_mask(users: np.ndarray, n_rows: int, val_ratio: float,
                  seed: int) -> np.ndarray:
    """Deterministic train/val assignment, hashed per user.

    Every row of one user lands on the same side, so validation measures unseen
    users instead of memorised histories. Without a user column the hash falls
    back to the row index.
    """
    if n_rows <= 0 or val_ratio <= 0.0:
        return np.zeros(max(n_rows, 0), dtype=bool)
    buckets = 10000
    hashed = np.zeros(n_rows, dtype=np.int64)
    got = min(len(users), n_rows)
    for i, v in enumerate(users[:got].tolist()):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            hashed[i] = (i * 2654435761 + seed) % buckets
        else:
            hashed[i] = zlib.crc32(str(v).encode("utf-8")) % buckets
    if got < n_rows:
        hashed[got:] = (np.arange(got, n_rows, dtype=np.int64) * 2654435761 + seed) % buckets
    mask = hashed < int(round(val_ratio * buckets))
    if not mask.any():
        mask[int(np.argmax(hashed))] = True
    if mask.all():
        mask[int(np.argmin(hashed))] = False
    return mask


def _iter_flat_chunks(
    pf: pq.ParquetFile,
    schema: SequenceSchema,
    batch_size: int,
    wanted_split: Optional[str],
    keep_rows: Optional[np.ndarray] = None,
    args=None,
    counters: Optional[Dict[str, int]] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray,
                    Optional[np.ndarray], np.ndarray]]:
    """Yield (items, ts, actions, offsets, targets, users) per parquet batch.

    Nothing is filtered here: `assemble_chunk` chooses the anchors from the full
    event stream, because a history may contain exposures and null-action
    events, and because one record can produce several rows.
    """
    cols = [schema.history_col]
    for c in (schema.history_ts_col, schema.history_actions_col,
              schema.target_col, schema.user_col, schema.split_col):
        if c:
            cols.append(c)

    click_value = float(getattr(args, "action_click_value", 1.0)) if args is not None else 1.0

    row_base = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        n = batch.num_rows
        if n == 0:
            continue
        hist = batch.column(schema.history_col)
        lens = _lengths_array(hist)
        items = _col_to_numpy(_flatten_list(hist))

        if len(items) != int(lens.sum()):
            items, lens = _lists_from_pylist(hist)
        offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)

        if schema.history_ts_col:
            tscol = batch.column(schema.history_ts_col)
            ts = _to_float_array(_col_to_numpy(_flatten_list(tscol)))
            if len(ts) != len(items):
                ts, _ = _lists_from_pylist(tscol, as_float=True)
            if len(ts) != len(items):
                ts = np.zeros(len(items), dtype=np.float64)
        else:
            ts = np.zeros(len(items), dtype=np.float64)
        ts = np.nan_to_num(_ts_to_seconds(ts), nan=0.0, posinf=0.0, neginf=0.0)

        target_raw = _col_to_numpy(batch.column(schema.target_col)) if schema.target_col else None

        act_codes: Optional[np.ndarray] = None
        if schema.history_actions_col:
            flat_acts: List[int] = []
            for row_acts in batch.column(schema.history_actions_col).to_pylist():
                if row_acts is None:
                    continue
                for a in row_acts:
                    flat_acts.append(_action_code(a, click_value))
            if len(flat_acts) != len(items):
                raise ValueError(
                    f"`{schema.history_actions_col}` holds {len(flat_acts)} actions but "
                    f"`{schema.history_col}` holds {len(items)} items; the two list "
                    "columns must be aligned"
                )
            act_codes = np.asarray(flat_acts, dtype=np.int64)
            if counters is not None:
                counters["unknown_actions"] += int((act_codes == ACTION_UNKNOWN).sum())
                counters["non_click_actions"] += int((act_codes == ACTION_OTHER).sum())

        user_raw = (
            _col_to_numpy(batch.column(schema.user_col))
            if schema.user_col else np.full(n, None, dtype=object)
        )

        mask = np.ones(n, dtype=bool)
        if schema.split_col and wanted_split is not None:
            split_vals = _col_to_numpy(batch.column(schema.split_col))
            mask &= np.asarray(
                [_split_matches(v, wanted_split) for v in split_vals.tolist()], dtype=bool
            )
        if keep_rows is not None:
            mask &= _row_slice(keep_rows, row_base, n)
        if not mask.all():
            sel = np.nonzero(mask)[0]
            row_of = np.repeat(np.arange(n, dtype=np.int64), lens)
            keep_flat = mask[row_of] if len(row_of) else row_of
            items = items[keep_flat]
            ts = ts[keep_flat]
            if act_codes is not None:
                act_codes = act_codes[keep_flat]
            if target_raw is not None:
                target_raw = target_raw[mask]
            user_raw = user_raw[mask]
            lens = lens[sel]
            offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)

        row_base += n
        yield items, ts, act_codes, offsets, target_raw, user_raw


def _iter_nested_chunks(
    pf: pq.ParquetFile,
    schema: SequenceSchema,
    batch_size: int,
    wanted_split: Optional[str],
    args,
    keep_rows: Optional[np.ndarray] = None,
    counters: Optional[Dict[str, int]] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray,
                    Optional[np.ndarray], np.ndarray]]:
    """Nested `list<struct<...>>` variant of `_iter_flat_chunks`.

    Same contract: every event of every row is passed through, targets come from
    `assemble_chunk`.
    """
    cols = [schema.nested_col]
    for c in (schema.target_col, schema.user_col, schema.split_col):
        if c:
            cols.append(c)

    click_value = float(args.action_click_value)

    row_base = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        rows = batch.to_pylist()
        if not rows:
            continue

        items_flat: List[Any] = []
        ts_flat: List[float] = []
        act_flat: List[int] = []
        lens: List[int] = []
        targets: Optional[List[Any]] = [] if schema.target_col else None
        users: List[Any] = []

        keep = _row_slice(keep_rows, row_base, len(rows)) if keep_rows is not None else None
        for i, row in enumerate(rows):
            if keep is not None and not keep[i]:
                continue
            if schema.split_col and not _split_matches(row.get(schema.split_col), wanted_split):
                continue

            interactions = row.get(schema.nested_col) or []
            n_events = 0
            for it in interactions:
                if not isinstance(it, dict):
                    items_flat.append(it)
                    ts_flat.append(0.0)
                    act_flat.append(ACTION_UNKNOWN)
                    n_events += 1
                    continue
                iid = _get_key(it, (schema.item_key,) + _ITEM_KEYS)
                if iid is None:
                    continue
                raw_ts = _get_key(it, (schema.ts_key,) + _TS_KEYS)
                raw_act = _get_key(it, (schema.action_key,) + _ACTION_KEYS)
                items_flat.append(iid)
                try:
                    ts_flat.append(float(raw_ts) if raw_ts is not None else 0.0)
                except (TypeError, ValueError):
                    ts_flat.append(0.0)
                act_flat.append(_action_code(raw_act, click_value))
                n_events += 1

            if n_events == 0:
                continue
            lens.append(n_events)
            if targets is not None:
                targets.append(row.get(schema.target_col))
            users.append(row.get(schema.user_col) if schema.user_col else None)

        row_base += len(rows)
        if not lens:
            continue

        lens_arr = np.asarray(lens, dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lens_arr)]).astype(np.int64)
        items = np.asarray(items_flat)
        ts = np.nan_to_num(
            _ts_to_seconds(np.asarray(ts_flat, dtype=np.float64)),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        act_codes = np.asarray(act_flat, dtype=np.int64)
        if counters is not None:
            counters["unknown_actions"] += int((act_codes == ACTION_UNKNOWN).sum())
            counters["non_click_actions"] += int((act_codes == ACTION_OTHER).sum())
        target_raw = np.asarray(targets, dtype=object) if targets is not None else None
        user_raw = np.asarray(users, dtype=object)
        yield items, ts, act_codes, offsets, target_raw, user_raw


# ---------------------------------------------------------------------------
# Phase 1/2: tokenised dataset
# ---------------------------------------------------------------------------

@dataclass
class TokenizedSplit:
    tokens: np.ndarray        # flat int32
    offsets: np.ndarray       # int64 [n_rows + 1]
    targets: np.ndarray       # int64 catalog indices
    users: np.ndarray         # raw user ids per row
    stats: Dict[str, Any]
    tokens_path: Optional[str] = None   # set when `tokens` is a memmap on disk

    def __post_init__(self):
        self.lengths = np.diff(self.offsets).astype(np.int64)

    def __len__(self) -> int:
        return len(self.targets)


def _sort_by_time(
    items: np.ndarray,
    ts: np.ndarray,
    act: Optional[np.ndarray],
    offsets: np.ndarray,
    stats: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Stable sort each record by timestamp.

    "Right to left" and "the gap to the previous event" only mean something in
    chronological order, so a record whose timestamps are out of order is sorted
    instead of silently producing a wrong history.  `rows_non_monotonic` counts
    the records that needed it.
    """
    n_rows = len(offsets) - 1
    n_events = len(ts)
    if n_rows <= 0 or n_events < 2:
        return items, ts, act
    delta = np.diff(ts)
    crossed = np.zeros(len(delta), dtype=bool)
    starts = offsets[1:-1]
    starts = starts[(starts > 0) & (starts < n_events)]   # empty records exist
    if len(starts):
        crossed[starts - 1] = True          # delta[i] spans two records
    bad = (delta < 0) & ~crossed
    if not bad.any():
        return items, ts, act
    row_of = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(offsets))
    bad_at = np.zeros(n_events, dtype=bool)     # `delta` is one shorter than `ts`
    bad_at[1:] = bad
    stats["rows_non_monotonic"] = int(np.unique(row_of[bad_at]).size)
    order = np.lexsort((np.arange(n_events, dtype=np.int64), ts, row_of))
    items = items[order]
    ts = ts[order]
    if act is not None:
        act = act[order]
    return items, ts, act


def _select_anchors(
    offsets: np.ndarray,
    act: Optional[np.ndarray],
    mode: str,
    anchor_gap: int,
    max_anchors: int,
    stats: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick the anchor (target) events of one chunk: (row index, event position).

    Anchors are walked right to left.  A click at position `p` is accepted when
    `previous_anchor - p >= anchor_gap`, so consecutive rows of one record share
    at most `window - anchor_gap` events of history; a rejected click does not
    consume the quota, the scan simply continues further left.
    """
    n_rows = len(offsets) - 1
    if n_rows <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    lens = np.diff(offsets).astype(np.int64)

    if act is None:
        # Without an action column nothing marks a click, so the only defensible
        # target is the last event and everything before it is history.
        rows = np.nonzero(lens > 0)[0].astype(np.int64)
        return rows, (offsets[1:] - 1)[rows].astype(np.int64)

    row_of = np.repeat(np.arange(n_rows, dtype=np.int64), lens)
    click_pos = np.nonzero(act == ACTION_CLICK)[0].astype(np.int64)
    click_count = np.bincount(row_of[click_pos], minlength=n_rows).astype(np.int64)
    stats["rows_without_click"] = int((click_count == 0).sum())
    if len(click_pos) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    click_row = row_of[click_pos]
    click_start = np.searchsorted(click_row, np.arange(n_rows, dtype=np.int64), side="left")
    rounds = 1 if mode == MODE_VAL else max(1, int(max_anchors))
    acc = np.full((rounds, n_rows), -1, dtype=np.int64)
    has = click_count > 0
    acc[0][has] = click_pos[click_start[has] + click_count[has] - 1]

    def _next_round(prev: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(click_pos, prev - int(anchor_gap), side="right") - 1
        ok = (prev >= 0) & (idx >= click_start) & has
        out = np.full(n_rows, -1, dtype=np.int64)
        out[ok] = click_pos[idx[ok]]
        return out

    for r in range(1, rounds):
        acc[r] = _next_round(acc[r - 1])
    if rounds > 1:
        # Would one more anchor have been possible?  Then the cap was binding.
        stats["rows_capped_by_anchors"] = int((_next_round(acc[-1]) >= 0).sum())

    per_row = (acc >= 0).sum(axis=0)
    stats["rows_with_single_anchor"] = int((per_row == 1).sum())
    stats["rows_gap_limited"] = int(
        ((per_row > 0) & (per_row < np.minimum(click_count, rounds))).sum()
    )

    rows = np.tile(np.arange(n_rows, dtype=np.int64), rounds)
    pos = acc.reshape(-1)
    ok = pos >= 0
    rows = rows[ok]
    pos = pos[ok]
    order = np.lexsort((pos, rows))
    return rows[order], pos[order]


def assemble_chunk(
    items_raw: np.ndarray,
    ts: np.ndarray,
    act_codes: Optional[np.ndarray],
    raw_offsets: np.ndarray,
    target_raw: Optional[np.ndarray],
    user_raw: np.ndarray,
    catalog: ItemCatalog,
    vocab: Vocab,
    feat_map: UserFeatureMap,
    gap_edges: np.ndarray,
    mode: str,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Vectorised [F.., SEP, (T, S1..SL), ...] construction for one parquet batch.

    Returns (tokens, offsets, targets, users, history_lengths, stats).  One input
    record can produce several rows (up to `--anchors_per_record`), and rows can
    come from different records, so nothing here assumes row i came from record i.
    """
    n_rows = len(raw_offsets) - 1
    stats: Dict[str, int] = {
        "rows_in": int(n_rows),
        "rows_kept": 0,
        "rows_proposed": 0,
        "rows_short": 0,
        "rows_unknown_history": 0,
        "rows_window_truncated": 0,
        "unknown_items": 0,
        "missing_target": 0,
        "rows_without_click": 0,
        "rows_capped_by_anchors": 0,
        "rows_gap_limited": 0,
        "rows_with_single_anchor": 0,
        "rows_non_monotonic": 0,
        "items_kept": 0,
        "feat_tokens": 0,
        "rows_not_in_feat_file": 0,
    }

    def _empty():
        return (
            np.empty(0, dtype=np.int32),
            np.zeros(1, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=object),
            np.empty(0, dtype=np.int64),
            stats,
        )

    if n_rows == 0:
        return _empty()

    # --- optional history filter: keep clicks only --------------------------
    if act_codes is not None and bool(getattr(args, "clicks_only", False)):
        row_of = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(raw_offsets))
        keep = act_codes == ACTION_CLICK
        items_raw = items_raw[keep]
        ts = ts[keep]
        act_codes = act_codes[keep]
        lens = np.bincount(row_of[keep], minlength=n_rows).astype(np.int64)
        raw_offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)

    items_raw, ts, act_codes = _sort_by_time(items_raw, ts, act_codes, raw_offsets, stats)

    # --- temporal token of every event, from the gap to its predecessor ------
    gap = np.zeros(len(ts), dtype=np.float64)
    if len(ts) > 1:
        gap[1:] = np.diff(ts)
    starts = raw_offsets[:-1][np.diff(raw_offsets) > 0]
    if len(starts):
        gap[starts] = 0.0
    gap = np.maximum(np.nan_to_num(gap, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    if gap_edges.size:
        bucket = np.digitize(gap, gap_edges).astype(np.int64)
    else:
        bucket = np.zeros(len(ts), dtype=np.int64)
    bucket = np.clip(bucket, 0, vocab.n_temporal - 1)
    temp_tok_all = (bucket + vocab.offset_temporal).astype(np.int32)
    clock_local = _clock_codes(ts) if vocab.n_clock else None

    # --- 1. anchors (what each row predicts) --------------------------------
    if target_raw is not None:
        # Explicit target column: one row per record, the given item is the label.
        anchor_row = np.arange(n_rows, dtype=np.int64)
        anchor_pos = raw_offsets[1:].copy()
        anchor_item: Optional[np.ndarray] = np.asarray(target_raw)
    elif mode == MODE_INFER:
        anchor_row = np.arange(n_rows, dtype=np.int64)
        anchor_pos = raw_offsets[1:].copy()
        anchor_item = None
    else:
        anchor_row, anchor_pos = _select_anchors(
            raw_offsets, act_codes, mode,
            int(args.anchor_gap), int(args.anchors_per_record), stats,
        )
        anchor_item = items_raw[anchor_pos] if len(anchor_pos) else None
    n_prop = int(len(anchor_row))
    stats["rows_proposed"] = n_prop
    if n_prop == 0:
        return _empty()

    # --- 2. window + minimum history ---------------------------------------
    window = int(args.window)
    row_start = raw_offsets[:-1][anchor_row]
    hist_len = np.minimum(anchor_pos - row_start, window)
    stats["rows_window_truncated"] = int((anchor_pos - row_start > window).sum())
    keep = hist_len >= int(args.min_history)
    if anchor_item is None and mode == MODE_INFER:
        keep = hist_len > 0                # inference keeps every record
    stats["rows_short"] = int((~keep).sum())
    if anchor_item is None:
        targets = np.full(n_prop, -1, dtype=np.int32)
    else:
        targets = catalog.lookup(anchor_item)
        stats["missing_target"] = int((keep & (targets < 0)).sum())
        keep = keep & (targets >= 0)
    sel = np.nonzero(keep)[0]
    if len(sel) == 0:
        return _empty()

    anchor_row = anchor_row[sel]
    hist_len = hist_len[sel]
    hist_end = anchor_pos[sel]
    hist_start = hist_end - hist_len
    targets = targets[sel].astype(np.int64)
    n_keep = int(len(sel))

    # --- 3. flatten the histories, drop items outside the catalog -----------
    total = int(hist_len.sum())
    if total == 0:
        return _empty()
    row_of_h = np.repeat(np.arange(n_keep, dtype=np.int64), hist_len)
    block_base = np.concatenate([[0], np.cumsum(hist_len)[:-1]]).astype(np.int64)
    flat = np.repeat(hist_start, hist_len) + (
        np.arange(total, dtype=np.int64) - block_base[row_of_h]
    )
    hist_idx = catalog.lookup(items_raw[flat])
    known = hist_idx >= 0
    stats["unknown_items"] = int((~known).sum())
    if not known.all():
        hist_idx = hist_idx[known]
        flat = flat[known]
        row_of_h = row_of_h[known]

    m = np.bincount(row_of_h, minlength=n_keep).astype(np.int64)
    nonempty = m > 0
    if not nonempty.all():
        remap = np.full(n_keep, -1, dtype=np.int64)
        remap[np.nonzero(nonempty)[0]] = np.arange(int(nonempty.sum()), dtype=np.int64)
        ev_row = remap[row_of_h]
        keep_ev = ev_row >= 0
        hist_idx = hist_idx[keep_ev]
        flat = flat[keep_ev]
        row_of_h = ev_row[keep_ev]
        targets = targets[nonempty]
        anchor_row = anchor_row[nonempty]
        m = m[nonempty]
        n_keep = int(nonempty.sum())
        stats["rows_unknown_history"] = int((~nonempty).sum())
        if n_keep == 0:
            return _empty()

    # --- 4. write [feature block, SEP, history] -----------------------------
    L = catalog.num_levels
    if vocab.num_levels != L:
        raise AssertionError(
            f"vocab has {vocab.num_levels} SID levels but the catalog has "
            f"{L}; rebuild the vocab from this catalog"
        )
    if vocab.num_features != feat_map.n_columns:
        raise AssertionError(
            f"vocab has {vocab.num_features} feature columns but the feature map "
            f"has {feat_map.n_columns}; rebuild the vocab from this feature file"
        )

    feat_row, feat_start, feat_len = feat_map.lookup(user_raw[anchor_row])
    slot = int(vocab.event_width)
    n_action = int(vocab.n_action)
    n_clock = int(vocab.n_clock)
    has_action = 1 if n_action else 0      # n_action is the vocab size (3)
    has_clock = 1 if n_clock else 0        # n_clock is the vocab size (169)
    if slot != 1 + has_action + has_clock + L:
        raise AssertionError(
            f"vocab.event_width={slot} is not 1 gap + {has_action} action + "
            f"{has_clock} clock + {L} SID tokens; the vocab does not match the catalog"
        )
    row_len = 1 + feat_len + m * slot
    offsets = np.concatenate([[0], np.cumsum(row_len)]).astype(np.int64)
    tokens = np.zeros(int(offsets[-1]), dtype=np.int32)
    tokens[offsets[:-1] + feat_len] = vocab.sep_id

    total_feat = int(feat_len.sum())
    if total_feat:
        row_of_f = np.repeat(np.arange(n_keep, dtype=np.int64), feat_len)
        fbase = np.concatenate([[0], np.cumsum(feat_len)[:-1]]).astype(np.int64)
        within_f = np.arange(total_feat, dtype=np.int64) - fbase[row_of_f]
        tokens[offsets[row_of_f] + within_f] = (
            feat_map.codes[feat_start[row_of_f] + within_f] + vocab.offset_features
        )

    row_m_base = np.concatenate([[0], np.cumsum(m)[:-1]]).astype(np.int64)
    pos_local = np.arange(len(hist_idx), dtype=np.int64) - row_m_base[row_of_h]
    base = offsets[row_of_h] + feat_len[row_of_h] + 1 + pos_local * slot
    tokens[base] = temp_tok_all[flat]

    # v2: the event's own action and clock go before its codes.  The SID block
    # stays last inside the slot, so "the last L tokens of a row are the last
    # event's codes" survives and train_digr.py needs no change.
    cursor = base + 1
    if has_action:
        if act_codes is not None:
            action_local = act_codes[flat]
        else:
            action_local = np.full(len(flat), ACTION_UNKNOWN, dtype=np.int64)
        action_local = np.where(
            action_local == ACTION_UNKNOWN, ACTION_UNKNOWN_TOKEN, action_local
        )
        tokens[cursor] = (action_local + vocab.offset_action).astype(np.int32)
        cursor = cursor + 1
    if has_clock:
        tokens[cursor] = (clock_local[flat] + vocab.offset_clock).astype(np.int32)
        cursor = cursor + 1

    sid_codes = catalog.sid_codes[hist_idx]
    for level in range(L):
        tokens[cursor + level] = sid_codes[:, level] + vocab.sid_offsets[level]

    if tokens.size and int(tokens.max()) >= vocab.total_size:
        raise AssertionError(
            f"token id {int(tokens.max())} exceeds vocab size {vocab.total_size}"
        )

    stats["feat_tokens"] = total_feat
    stats["rows_not_in_feat_file"] = int((feat_row == feat_map.n_rows).sum())
    stats["rows_kept"] = int(n_keep)
    stats["items_kept"] = int(m.sum())
    return tokens, offsets, targets, user_raw[anchor_row], m, stats


def build_tokenized_split(
    path: str,
    schema: SequenceSchema,
    catalog: ItemCatalog,
    vocab: Vocab,
    feat_map: UserFeatureMap,
    gap_edges: np.ndarray,
    args,
    logger: logging.Logger,
    wanted_split: Optional[str] = None,
    mode: str = MODE_TRAIN,
    row_limit: Optional[int] = None,
    keep_rows: Optional[np.ndarray] = None,
    stream_dir: Optional[str] = None,
) -> TokenizedSplit:
    """Tokenise one file/split into a TokenizedSplit.

    `mode` (train / val / infer) decides the anchors of each record, see the
    module docstring.  `keep_rows` keeps only the given global row indices (used
    to carve a per-user train/val split out of a single file).  `stream_dir`
    spills the token stream straight to `<stream_dir>/tokens.npy` instead of
    building the whole int32 array in RAM first.
    """
    counters: Dict[str, int] = {"unknown_actions": 0, "non_click_actions": 0}
    pf = pq.ParquetFile(path)
    if schema.nested_col is not None:
        chunk_iter = _iter_nested_chunks(
            pf, schema, args.read_batch_size, wanted_split, args,
            keep_rows=keep_rows, counters=counters,
        )
    else:
        chunk_iter = _iter_flat_chunks(
            pf, schema, args.read_batch_size, wanted_split,
            keep_rows=keep_rows, args=args, counters=counters,
        )

    tok_file = None
    tok_path: Optional[str] = None
    tok_raw: Optional[str] = None
    if stream_dir is not None:
        os.makedirs(stream_dir, exist_ok=True)
        tok_path = os.path.join(stream_dir, "tokens.npy")
        tok_raw = tok_path + ".raw"
        tok_file = open(tok_raw, "wb")

    tokens_parts: List[np.ndarray] = []
    offsets_parts: List[np.ndarray] = []
    targets_parts: List[np.ndarray] = []
    users_parts: List[np.ndarray] = []

    agg: Dict[str, int] = {}
    base = 0
    raw_rows = 0
    kept_rows = 0
    t0 = time.time()

    for bi, chunk in enumerate(chunk_iter):
        items, ts, act, raw_offsets, target_raw, user_raw = chunk
        raw_rows += len(raw_offsets) - 1
        tok, off, tgt, usr, _, st = assemble_chunk(
            items, ts, act, raw_offsets, target_raw, user_raw,
            catalog, vocab, feat_map, gap_edges, mode, args,
        )
        for key, value in st.items():
            agg[key] = agg.get(key, 0) + int(value)
        if len(tok):
            if tok_file is not None:
                tok.astype(np.int32, copy=False).tofile(tok_file)
            else:
                tokens_parts.append(tok)
            offsets_parts.append(off + base)
            targets_parts.append(tgt)
            users_parts.append(usr)
            base += len(tok)
            kept_rows += len(tgt)

        if bi % 25 == 0:
            elapsed = max(1e-6, time.time() - t0)
            logger.info(
                f"    tokenising: {human_int(raw_rows)} rows read, "
                f"{human_int(kept_rows)} kept, {human_int(base)} tokens "
                f"({raw_rows / elapsed:,.0f} rows/s)"
            )

        if row_limit is not None and raw_rows >= row_limit:
            break

    if tok_file is not None:
        tok_file.close()
        tok_file = None

    if kept_rows == 0:
        for p in (tok_path, tok_raw):
            if p is not None and os.path.isfile(p):
                os.remove(p)
        raise ValueError(
            f"no usable sequences in {path}: no event could become a target. "
            f"Need {args.min_history} events of history before a click "
            f"(window={args.window}, anchors/record={args.anchors_per_record}, "
            f"anchor_gap={args.anchor_gap}). Check the sequence schema, the action "
            "column (only action_type == --action_click_value is a target) and the "
            "catalog coverage. If the sequences store 1-based catalog indices (as "
            "`preprocess_sequence_cli.py` does) instead of raw item ids, remap them "
            "to item ids before running."
        )

    if tok_path is not None and tok_raw is not None:
        # Raw stream -> one regular .npy (header + payload).  `open_memmap` on a
        # "*.npy" name would leave a sidecar "*.npy.raw" next to it.
        header = {"descr": "<i4", "fortran_order": False, "shape": (base,)}
        with open(tok_path, "wb") as fout:
            np.lib.format.write_array_header_1_0(fout, header)
            with open(tok_raw, "rb") as fin:
                shutil.copyfileobj(fin, fout, length=1 << 24)
        os.remove(tok_raw)
        tokens = np.load(tok_path, mmap_mode="r")
    else:
        tokens = np.concatenate(tokens_parts)
    offsets = np.concatenate([offsets_parts[0]] + [o[1:] for o in offsets_parts[1:]])
    targets = np.concatenate(targets_parts)
    users = np.concatenate(users_parts)

    stats = {
        "rows_read": raw_rows,
        "rows_kept": kept_rows,
        "anchors_per_record": kept_rows / max(1, raw_rows),
        "tokens": int(len(tokens)),
        "avg_items": agg.get("items_kept", 0) / max(1, kept_rows),
        "rows_proposed": agg.get("rows_proposed", 0),
        "rows_short": agg.get("rows_short", 0),
        "rows_unknown_history": agg.get("rows_unknown_history", 0),
        "rows_window_truncated": agg.get("rows_window_truncated", 0),
        "rows_gap_limited": agg.get("rows_gap_limited", 0),
        "rows_capped_by_anchors": agg.get("rows_capped_by_anchors", 0),
        "rows_with_single_anchor": agg.get("rows_with_single_anchor", 0),
        "rows_without_click": agg.get("rows_without_click", 0),
        "rows_missing_target": agg.get("missing_target", 0),
        "rows_non_monotonic": agg.get("rows_non_monotonic", 0),
        "unknown_items": agg.get("unknown_items", 0),
        "unknown_actions": counters["unknown_actions"],
        "non_click_actions": counters["non_click_actions"],
        "feat_tokens": agg.get("feat_tokens", 0),
        "rows_not_in_feat_file": agg.get("rows_not_in_feat_file", 0),
    }
    logger.info(
        f"  + {os.path.basename(path)}: {human_int(kept_rows)} rows from "
        f"{human_int(raw_rows)} records ({stats['anchors_per_record']:.2f}/record), "
        f"{human_int(len(tokens))} tokens, avg {stats['avg_items']:.1f} events/row, "
        f"window-capped {human_int(stats['rows_window_truncated'])} "
        f"({time.time() - t0:.1f}s)"
    )
    logger.info(
        f"    limits: anchors {human_int(stats['rows_capped_by_anchors'])}, "
        f"gap {human_int(stats['rows_gap_limited'])}, "
        f"min_history {human_int(stats['rows_short'])}, "
        f"all-history-unknown {human_int(stats['rows_unknown_history'])}, "
        f"no-click {human_int(stats['rows_without_click'])}, "
        f"target-not-in-catalog {human_int(stats['rows_missing_target'])}, "
        f"unknown history items {human_int(stats['unknown_items'])}, "
        f"out-of-order records {human_int(stats['rows_non_monotonic'])}"
    )
    if stats["feat_tokens"]:
        logger.info(
            f"    features: {human_int(stats['feat_tokens'])} tokens "
            f"({human_int(stats['rows_not_in_feat_file'])} rows of users "
            "missing from --user_feat_path, each all-UNK)"
        )
    if tokens_parts:
        tokens_parts.clear()
    return TokenizedSplit(
        tokens=tokens, offsets=offsets, targets=targets, users=users, stats=stats,
        tokens_path=tok_path,
    )


# ---------------------------------------------------------------------------
# Data plumbing for train / infer
# ---------------------------------------------------------------------------

def _collect_users(path: str, col: Optional[str], limit: Optional[int],
                   batch_size: int = 65536) -> Tuple[np.ndarray, bool]:
    if col is None:
        return np.empty(0, dtype=np.int64), True
    pf = pq.ParquetFile(path)
    parts: List[np.ndarray] = []
    n = 0
    for batch in pf.iter_batches(columns=[col], batch_size=batch_size):
        parts.append(_col_to_numpy(batch.column(col)))
        n += batch.num_rows
        if limit is not None and n >= limit:
            break
    if not parts:
        return np.empty(0, dtype=np.int64), True
    arr = np.concatenate(parts)
    if limit is not None:
        arr = arr[: int(limit)]        # the last batch usually overshoots
    return normalize_ids(arr)


def save_split(split: "TokenizedSplit", out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, "tokens.npy")
    already = split.tokens_path is not None and (
        os.path.abspath(split.tokens_path) == os.path.abspath(target)
    )
    if not already:      # streaming build already wrote tokens.npy in place
        np.save(target, split.tokens.astype(np.int32, copy=False))
    np.save(os.path.join(out_dir, "offsets.npy"), split.offsets.astype(np.int64, copy=False))
    np.save(os.path.join(out_dir, "targets.npy"), split.targets.astype(np.int64, copy=False))
    np.save(os.path.join(out_dir, "users.npy"), np.asarray(split.users, dtype=object))
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({k: (float(v) if isinstance(v, float) else int(v))
                   for k, v in split.stats.items()}, f, indent=2)


def load_split(data_dir: str, name: str) -> "TokenizedSplit":
    """Load `data_dir/<name>/` back into a TokenizedSplit (used by training/inference).

    `tokens.npy` is memory-mapped: DataLoader workers (Windows spawn) share the
    pages instead of copying multi-GB arrays per worker.
    """
    d = os.path.join(data_dir, name)
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"split '{name}' not found in {data_dir} (run preprocess_digr.py first)"
        )
    missing = [f for f in SPLIT_FILES if not os.path.isfile(os.path.join(d, f))]
    if missing:
        raise FileNotFoundError(
            f"split '{name}' in {data_dir} is incomplete (missing: {', '.join(missing)}); "
            "re-run preprocess_digr.py"
        )
    stats = {}
    stats_path = os.path.join(d, "stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    tokens_path = os.path.join(d, "tokens.npy")
    return TokenizedSplit(
        tokens=np.load(tokens_path, mmap_mode="r"),
        offsets=np.load(os.path.join(d, "offsets.npy")),
        targets=np.load(os.path.join(d, "targets.npy")),
        users=np.load(os.path.join(d, "users.npy"), allow_pickle=True),
        stats=stats,
        tokens_path=tokens_path,
    )


def split_exists(data_dir: str, name: str) -> bool:
    """True only when every file of the split is present (a half-written split is not)."""
    d = os.path.join(data_dir, name)
    return os.path.isdir(d) and all(os.path.isfile(os.path.join(d, f)) for f in SPLIT_FILES)


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------

def limit_levels(catalog: "ItemCatalog", n_levels: Optional[int],
                 logger: logging.Logger) -> "ItemCatalog":
    """Keep only the first `n_levels` SID levels (drops the unique-id rank code)."""
    if n_levels is None or n_levels <= 0 or n_levels >= catalog.num_levels:
        return catalog
    logger.info(
        f"  + keeping {n_levels}/{catalog.num_levels} SID levels "
        f"(levels {n_levels + 1}..{catalog.num_levels} are dropped as non-semantic)"
    )
    codes = np.ascontiguousarray(catalog.sid_codes[:, :n_levels])
    return ItemCatalog(catalog.item_ids, catalog.is_int, codes, prepared=True)


def catalog_collision_stats(catalog: "ItemCatalog") -> Dict[str, Any]:
    """How many items share an identical SID, over *every* level kept.

    v2 catalogues carry no `--add_unique_id` suffix, so the whole code is the
    SID and any duplicate is a pair of items the model can never tell apart.
    The share is reported because it is exactly the fraction of labels whose
    rank is a coin flip, and the headline table scores exact identity.
    """
    codes = np.asarray(catalog.sid_codes)
    n_items = int(codes.shape[0])
    if codes.size == 0:
        return {"levels": 0, "distinct_sids": 0, "items_sharing_a_sid": 0,
                "collision_share": 0.0, "max_items_per_sid": 0}
    # unique(axis=0) sorts whole rows, so no mixed-radix packing and no overflow
    _, counts = np.unique(codes, axis=0, return_counts=True)
    dup = int(counts[counts > 1].sum()) if counts.size else 0
    return {
        "levels": int(codes.shape[1]),
        "distinct_sids": int(counts.size),
        "items_sharing_a_sid": dup,
        "collision_share": dup / max(1, n_items),
        "max_items_per_sid": int(counts.max()) if counts.size else 0,
    }


def _load_catalog(path: str, args, logger: logging.Logger) -> "ItemCatalog":
    if path.lower().endswith((".json", ".jsonl")):
        return load_catalog_from_json(path, args, logger)
    return load_catalog_from_parquet(path, args, logger)


# ---------------------------------------------------------------------------
# Preprocessing entry point
# ---------------------------------------------------------------------------

def _wipe_output_dir(out_dir: str) -> None:
    """--overwrite: drop the previous run before any log file is opened inside it."""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)


def _check_output_dir(out_dir: str, overwrite: bool) -> None:
    if overwrite or not os.path.isdir(out_dir):
        return
    existing = [f for f in ("vocab.json", MANIFEST_NAME, "train", "val")
                if os.path.exists(os.path.join(out_dir, f))]
    if existing:
        raise SystemExit(
            f"{out_dir} already contains preprocessed data ({', '.join(existing)}); "
            "pass --overwrite to replace it"
        )


def preprocess(args, logger: logging.Logger) -> Dict[str, Any]:
    t_all = time.time()
    out_dir = args.output_dir
    _check_output_dir(out_dir, args.overwrite)
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. catalog + user features + token layout --------------------------
    log_banner(logger, "DiGR PREPROCESSING")
    log_card(
        logger,
        "inputs",
        {
            "Sequence": args.sequence_path,
            "Val sequence": args.val_sequence_path or "(split from sequence)",
            "Infer sequence": args.infer_sequence_path or "-",
            "Catalog": args.catalog_path,
            "Output": out_dir,
        },
    )

    catalog = _load_catalog(args.catalog_path, args, logger)
    catalog = limit_levels(catalog, args.num_sid_levels, logger)
    collisions = catalog_collision_stats(catalog)
    logger.info(
        f"  + SID: {human_int(collisions['distinct_sids'])} distinct codes over "
        f"{human_int(catalog.size)} items x {collisions['levels']} levels; "
        f"{human_int(collisions['items_sharing_a_sid'])} items share a code "
        f"({collisions['collision_share']:.4%}), max "
        f"{human_int(collisions['max_items_per_sid'])}/group"
    )
    if collisions["collision_share"] > 0.01:
        logger.warning(
            f"  ! {collisions['collision_share']:.2%} of the catalogue shares a "
            "SID, so that share of labels can never be ranked exactly; the "
            "headline table scores exact identity"
        )

    seq_schema = detect_sequence_schema(args.sequence_path, args)
    logger.info(f"  > sequence schema: {describe_schema(seq_schema)}")
    sources = [(args.sequence_path, seq_schema.user_col, args.max_rows)]

    val_schema = None
    if args.val_sequence_path:
        val_schema = detect_sequence_schema(args.val_sequence_path, args)
        logger.info(f"  > val schema: {describe_schema(val_schema)}")
        sources.append((args.val_sequence_path, val_schema.user_col, args.max_val_rows))

    infer_schema = None
    if args.infer_sequence_path:
        infer_schema = detect_sequence_schema(args.infer_sequence_path, args)
        logger.info(f"  > infer schema: {describe_schema(infer_schema)}")
        sources.append((args.infer_sequence_path, infer_schema.user_col, args.max_rows))

    if args.user_feat_path:
        feat_map = build_user_feature_map(args.user_feat_path, args, logger)
    else:
        feat_map = UserFeatureMap.empty()
        logger.info("  + no --user_feat_path: rows start with SEP, no user context")

    vocab = build_vocab(catalog, feat_map, args)
    gap_edges = build_gap_edges(args.n_temporal_bins, args.max_gap_seconds)
    logger.info(
        f"  + vocabulary: total={human_int(vocab.total_size)} "
        f"(special={vocab.n_special}, SID={vocab.sid_sizes} @ {vocab.sid_offsets}, "
        f"temporal={vocab.n_temporal} @ {vocab.offset_temporal}, "
        f"action={vocab.n_action} @ {vocab.offset_action}, "
        f"clock={vocab.n_clock} @ {vocab.offset_clock}, "
        f"features={vocab.feature_sizes} @ {vocab.offset_features}, "
        f"event_width={vocab.event_width})"
    )

    # --- 2. persist the static artefacts ----------------------------------
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab.to_dict(), f, indent=2)
    catalog.save(out_dir)
    # train_digr.py still loads a UserMap; the user block is now the feature
    # block, so it stays empty on purpose (no user id is tokenised any more).
    UserMap(np.empty(0, dtype=np.int64), True).save(out_dir)
    feat_map.save(out_dir)
    np.save(os.path.join(out_dir, "gap_edges.npy"), gap_edges)
    logger.info(
        f"  + wrote vocab.json / catalog_* / user_feat_meta.json / "
        f"gap_edges.npy -> {out_dir}"
    )

    # --- 3. tokenise the splits -------------------------------------------
    def build(path, schema, wanted, limit, mode=MODE_TRAIN,
              keep_rows=None, dir_name=None):
        # Streaming: the token stream lands directly in <out_dir>/<dir_name>/tokens.npy
        return build_tokenized_split(
            path, schema, catalog, vocab, feat_map, gap_edges, args, logger,
            wanted_split=wanted, mode=mode, row_limit=limit,
            keep_rows=keep_rows,
            stream_dir=os.path.join(out_dir, dir_name) if dir_name else None,
        )

    splits: Dict[str, int] = {}
    stats: Dict[str, Any] = {}

    if args.val_sequence_path:
        train_split = build(args.sequence_path, seq_schema, None, args.max_rows,
                            dir_name="train")
        val_split = build(args.val_sequence_path, val_schema, None, args.max_val_rows,
                          mode=MODE_VAL, dir_name="val")
    elif seq_schema.split_col is not None:
        train_split = build(args.sequence_path, seq_schema, "train", args.max_rows,
                            dir_name="train")
        val_split = build(args.sequence_path, seq_schema, "val", args.max_val_rows,
                          mode=MODE_VAL, dir_name="val")
    else:
        # Single file without a split column: hash users, not rows, so a user's
        # whole history stays on one side (the two passes filter rows before
        # assembly, so nothing is tokenised twice).
        total_rows = int(pq.ParquetFile(args.sequence_path).metadata.num_rows)
        if args.max_rows is not None:
            total_rows = min(total_rows, int(args.max_rows))
        if total_rows < 2:
            raise SystemExit("need at least 2 sequences to create a train/val split")
        users_for_split, _ = _collect_users(
            args.sequence_path, seq_schema.user_col, total_rows, args.read_batch_size
        )
        val_mask = _val_row_mask(
            users_for_split, total_rows, float(args.val_ratio), int(args.seed)
        )
        del users_for_split
        n_val = int(val_mask.sum())
        logger.info(
            f"  + random split by user: {human_int(total_rows - n_val)} train / "
            f"{human_int(n_val)} val rows ({args.val_ratio:.1%})"
        )
        train_split = build(args.sequence_path, seq_schema, None, args.max_rows,
                            keep_rows=~val_mask, dir_name="train")
        val_split = build(args.sequence_path, seq_schema, None, args.max_rows,
                          mode=MODE_VAL, keep_rows=val_mask, dir_name="val")

    if len(train_split) == 0:
        raise SystemExit("training split is empty; check --min_history and the catalog")
    if len(val_split) == 0:
        raise SystemExit("validation split is empty; lower --val_ratio or pass --val_sequence_path")

    save_split(train_split, os.path.join(out_dir, "train"))
    save_split(val_split, os.path.join(out_dir, "val"))
    splits["train"] = int(len(train_split))
    splits["val"] = int(len(val_split))
    stats["train"] = train_split.stats
    stats["val"] = val_split.stats

    if infer_schema is not None:
        # No anchor: inference keeps the whole windowed history and has no label.
        infer_split = build(args.infer_sequence_path, infer_schema, None,
                            args.max_rows, mode=MODE_INFER, dir_name="infer")
        if len(infer_split):
            save_split(infer_split, os.path.join(out_dir, "infer"))
            splits["infer"] = int(len(infer_split))
            stats["infer"] = infer_split.stats
        else:
            logger.warning("  ! inference split is empty; nothing written to infer/")

    # --- 4. manifest -------------------------------------------------------
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "sources": {
            "sequence_path": os.path.abspath(args.sequence_path),
            "val_sequence_path": os.path.abspath(args.val_sequence_path)
            if args.val_sequence_path else None,
            "infer_sequence_path": os.path.abspath(args.infer_sequence_path)
            if args.infer_sequence_path else None,
            "catalog_path": os.path.abspath(args.catalog_path),
            "user_feat_path": os.path.abspath(args.user_feat_path)
            if args.user_feat_path else None,
        },
        "sequence_schema": describe_schema(seq_schema),
        "splits": splits,
        "split_stats": stats,
        "catalog_size": int(catalog.size),
        "num_levels": int(catalog.num_levels),
        "catalog_collisions": collisions,
        "clicks_only": bool(args.clicks_only),
        "action_click_value": float(args.action_click_value),
        "anchors_per_record": int(args.anchors_per_record),
        "anchor_gap": int(args.anchor_gap),
        "history_actions_col": seq_schema.history_actions_col,
        "vocab_size": int(vocab.total_size),
        "sid_sizes": list(vocab.sid_sizes),
        "sid_offsets": list(vocab.sid_offsets),
        "offset_temporal": int(vocab.offset_temporal),
        "offset_features": int(vocab.offset_features),
        "feature_sizes": [int(s) for s in vocab.feature_sizes],
        "feature_offsets": [int(o) for o in vocab.feature_offsets],
        "n_temporal_bins": int(vocab.n_temporal),
        "n_action": int(vocab.n_action),
        "n_clock": int(vocab.n_clock),
        "offset_action": int(vocab.offset_action),
        "offset_clock": int(vocab.offset_clock),
        "event_width": int(vocab.event_width),
        "action_token_order": ["exposure", "click", "unknown"],
        "n_users": int(feat_map.n_rows),
        "user_features": feat_map.meta(),
        "max_feat_tokens": int(feat_map.max_len),
        "max_seq_len": int(args.window),      # kept for train_digr.py
        "window": int(args.window),
        "min_history": int(args.min_history),
        "max_gap_seconds": float(args.max_gap_seconds),
        "val_ratio": float(args.val_ratio),
        "seed": int(args.seed),
        "elapsed_seconds": round(time.time() - t_all, 1),
    }
    with open(os.path.join(out_dir, MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log_card(
        logger,
        "summary",
        {
            "Catalog": f"{human_int(catalog.size)} items x {catalog.num_levels} levels",
            "Collisions": f"{human_int(collisions['items_sharing_a_sid'])} dup items, "
                          f"{collisions['collision_share']:.4%} of the catalogue, "
                          f"max {collisions['max_items_per_sid']}/group",
            "Clicks only": str(bool(args.clicks_only)),
            "Tokens": human_int(vocab.total_size),
            "Event width": f"{vocab.event_width} tokens (1 gap + "
                            f"{1 if vocab.n_action else 0} action + "
                            f"{1 if vocab.n_clock else 0} clock + "
                            f"{vocab.num_levels} SID)",
            "Users": f"{human_int(feat_map.n_rows)} x "
                     f"{feat_map.n_columns} feature cols",
            "Rows / record": f"{stats['train']['anchors_per_record']:.2f} train, "
                             f"{stats['val']['anchors_per_record']:.2f} val",
            "Train / val": f"{human_int(splits['train'])} / {human_int(splits['val'])}",
            "Infer": human_int(splits.get("infer", 0)),
            "Window": f"{args.window} events",
            "Anchors": f"k={args.anchors_per_record}, gap>={args.anchor_gap} events",
            "Output": os.path.abspath(out_dir),
            "Elapsed": f"{time.time() - t_all:.1f}s",
        },
    )
    logger.info(f"  + manifest -> {os.path.join(out_dir, MANIFEST_NAME)}")
    logger.info("  next: python train_digr.py --mode train --data_dir " + out_dir)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _none_if_blank(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DiGR preprocessing: parquet sequences + SID catalog -> digr_data/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--sequence_path", type=str, default=DEFAULT_SEQUENCE_PATH)
    p.add_argument("--val_sequence_path", type=str, default=None)
    p.add_argument("--infer_sequence_path", type=str, default=None,
                   help="optional extra file tokenised WITHOUT leave-one-out "
                        "(no targets) and written to <output_dir>/infer/")
    p.add_argument("--catalog_path", "--candidates_path", dest="catalog_path",
                   type=str, default=DEFAULT_CATALOG_PATH,
                   help="item catalog: parquet (item_id + SID columns) or the "
                        "RQ-VAE json {item_id: [codes]}")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing non-empty --output_dir")

    # sequence schema overrides
    p.add_argument("--user_col", type=str, default=None)
    p.add_argument("--seq_col", type=str, default=None, help="list<struct> interaction column")
    p.add_argument("--history_col", type=str, default=None, help="flat list<int> history column")
    p.add_argument("--history_ts_col", type=str, default=None)
    p.add_argument("--history_actions_col", type=str, default=None,
                   help="flat list column of action types aligned with "
                        "--history_col (auto-detected as history_actions/actions)")
    p.add_argument("--target_col", type=str, default=None)
    p.add_argument("--split_col", type=str, default=None)
    p.add_argument("--item_key", type=str, default=None)
    p.add_argument("--ts_key", type=str, default=None)
    p.add_argument("--action_key", type=str, default=None)
    p.add_argument("--action_click_value", type=float, default=1.0,
                   help="value of `action_type` that counts as a click")
    p.add_argument("--clicks_only", action=argparse.BooleanOptionalAction, default=False,
                   help="keep only action_type==--action_click_value events in the "
                        "history (default keeps every event: clicks, exposures and "
                        "null actions all count as history, only clicks can be a "
                        "target)")

    # catalog schema overrides
    p.add_argument("--catalog_item_col", type=str, default=None)
    p.add_argument("--catalog_sid_cols", type=str, nargs="+", default=None)
    p.add_argument("--codebook_size", type=int, default=None,
                   help="fixed codebook size per level (default: max code + 1)")
    p.add_argument("--code_offset", type=int, default=0,
                   help="subtract this from every SID code read from a .json "
                        "catalog (use 1 for 1-based code dumps)")
    p.add_argument("--num_sid_levels", type=int, default=DEFAULT_NUM_SID_LEVELS,
                   help="keep only the first N SID levels; 0 (the default) keeps "
                        "every level, i.e. the `--add_unique_id` collision rank is "
                        "tokenised as the last SID level. Use 4 to drop it.")

    # tokenisation
    p.add_argument("--window", "--max_seq_len", dest="window", type=int,
                   default=DEFAULT_WINDOW,
                   help="events of history kept before each anchor (the most "
                        "recent ones)")
    p.add_argument("--min_history", type=int, default=DEFAULT_MIN_HISTORY,
                   help="events of history a row needs to be kept")
    p.add_argument("--anchors_per_record", type=int, default=DEFAULT_ANCHORS_PER_RECORD,
                   help="max rows one record may produce (its most recent clicks)")
    p.add_argument("--anchor_gap", type=int, default=DEFAULT_ANCHOR_GAP,
                   help="minimum distance in events between two accepted anchors "
                        "of one record; skipped clicks do not count against "
                        "--anchors_per_record")
    p.add_argument("--n_temporal_bins", type=int, default=DEFAULT_N_TEMPORAL_BINS,
                   help="log-spaced gap bins; v2 defaults to 128 where v1 used 64")
    p.add_argument("--add_action", action=argparse.BooleanOptionalAction,
                   default=DEFAULT_ADD_ACTION,
                   help="emit a per-event action token (0 exposure, 1 click, "
                        "2 unknown) between the gap and the codes")
    p.add_argument("--add_clock", action=argparse.BooleanOptionalAction,
                   default=DEFAULT_ADD_CLOCK,
                   help="emit a per-event hour-of-day x day-of-week token (168 "
                        "slots, UTC) after the action token; costs one token "
                        "per event")
    p.add_argument("--max_gap_seconds", type=float, default=DEFAULT_MAX_GAP_SECONDS)
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--max_val_rows", type=int, default=None)
    p.add_argument("--user_feat_path", type=str, default=None,
                   help="optional parquet of static user attributes (one row per "
                        "user, e.g. data/user_feat.parquet); every non-id column "
                        "becomes a per-user feature block in front of the history")
    p.add_argument("--user_feat_id_col", type=str, default=None,
                   help="user id column of --user_feat_path (auto-detected)")
    p.add_argument("--user_feat_cols", type=str, nargs="+", default=None,
                   help="columns of --user_feat_path to tokenise (default: every "
                        "column except the id column, in file order)")
    p.add_argument("--max_feat_len", type=int, default=DEFAULT_MAX_FEAT_LEN,
                   help="max slots one LIST feature column may use (0 = the longest "
                        "list actually seen); longer lists are truncated")
    p.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    p.add_argument("--read_batch_size", type=int, default=DEFAULT_READ_BATCH_SIZE)
    p.add_argument("--seed", type=int, default=SEED)

    args = p.parse_args(argv)
    args.user_col = _none_if_blank(args.user_col)
    args.seq_col = _none_if_blank(args.seq_col)
    args.history_col = _none_if_blank(args.history_col)
    args.history_ts_col = _none_if_blank(args.history_ts_col)
    args.history_actions_col = _none_if_blank(args.history_actions_col)
    args.target_col = _none_if_blank(args.target_col)
    args.split_col = _none_if_blank(args.split_col)
    args.item_key = _none_if_blank(args.item_key)
    args.ts_key = _none_if_blank(args.ts_key)
    args.action_key = _none_if_blank(args.action_key)
    args.catalog_item_col = _none_if_blank(args.catalog_item_col)
    args.val_sequence_path = _none_if_blank(args.val_sequence_path)
    args.infer_sequence_path = _none_if_blank(args.infer_sequence_path)
    args.user_feat_path = _none_if_blank(args.user_feat_path)
    args.user_feat_id_col = _none_if_blank(args.user_feat_id_col)

    if args.n_temporal_bins < 1:
        p.error("--n_temporal_bins must be >= 1")
    if args.window < 1:
        p.error("--window must be >= 1")
    if args.min_history < 1:
        p.error("--min_history must be >= 1")
    if args.anchors_per_record < 1:
        p.error("--anchors_per_record must be >= 1")
    if args.anchor_gap < 0:
        p.error("--anchor_gap must be >= 0")
    if not 0.0 <= args.val_ratio < 1.0:
        p.error("--val_ratio must be in [0, 1)")
    if args.max_feat_len < 0:
        p.error("--max_feat_len must be >= 0 (0 = longest list seen)")
    if args.read_batch_size < 1:
        p.error("--read_batch_size must be >= 1")
    if args.max_gap_seconds <= 0:
        p.error("--max_gap_seconds must be > 0")
    return args


def _validate_paths(args) -> None:
    if not os.path.exists(args.sequence_path):
        raise SystemExit(
            f"sequence file not found: {args.sequence_path}\n"
            "expected parquet with either (user_id, seq=list<struct<item_id,"
            "action_type,timestamp>>) or (history_items, [history_timestamps,] target)"
        )
    if not os.path.exists(args.catalog_path):
        raise SystemExit(
            f"catalog file not found: {args.catalog_path}\n"
            "expected parquet (item_id + SID columns) or the RQ-VAE json "
            "mapping {item_id: [codes]}"
        )
    if args.val_sequence_path and not os.path.exists(args.val_sequence_path):
        raise SystemExit(f"validation file not found: {args.val_sequence_path}")
    if args.infer_sequence_path and not os.path.exists(args.infer_sequence_path):
        raise SystemExit(f"inference file not found: {args.infer_sequence_path}")
    if args.user_feat_path and not os.path.exists(args.user_feat_path):
        raise SystemExit(f"user feature file not found: {args.user_feat_path}")


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    _validate_paths(args)
    if args.overwrite:
        _wipe_output_dir(args.output_dir)
    logger = setup_logger(os.path.join(args.output_dir, "logs"))
    try:
        preprocess(args, logger)
    except SystemExit:
        raise
    except Exception:
        logger.exception("fatal error")
        raise


if __name__ == "__main__":
    main()
