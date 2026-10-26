#!/usr/bin/env python3
"""Score the Stage-1 item embedding against the real next click.

Why this file exists
--------------------
Every number in `sid_quality_eda.ipynb` / `data_peak.ipynb` is scored against the
embedding's own nearest neighbours: cell 24 builds `true_neighbors` with
`index.search` over the same vectors, so "depth-1 recall 71.158%" means *the
depth-1 bucket contains 71% of the embedding's own top-10 items*. The embedding
is grading its own homework, and the codebook is being judged by the geometry
that produced it.

This script asks the only question that cannot be answered that way: does the
geometry predict what a user actually clicked next? Same 98,304 validation rows,
same `val/targets.npy` the baselines and DiGR are scored on, exact cosine over
the whole 4,783,154 item catalog.

Probes
------
`knn w`   query = mean of the embeddings of the last w clicked items, renormalised.
          w=1 is the single most recent item (comparable to co-click window 1),
          w=5 is comparable to the published co-click window-5 row.
`sid d`   the depth-d SID prefix bucket of the last clicked item, as an index,
          then ranked inside the bucket by cosine to the same query. Reported as
          candidate recall (is the target in the bucket at all) and R@k after
          the cosine rerank. This is the non-circular version of the notebook's
          prefix table.

Reading the answer
------------------
R@10 >= 5%     the embedding knows the next click and the 4-by-256 code is
               throwing it away. The SID is the wrong interface; serve
               FAISS + reranker and redesign TIGER before writing it.
2.5% - 4.5%    embedding, co-click count table (3.6011%) and DiGR (2.5940%) all
               know the same amount. The SID is exonerated and TIGER proceeds as
               specified.
R@10 < 2%      pairwise item geometry does not encode intent; the behaviour is
               only in the HSTU sequence tower. The last-item query is the wrong
               probe and this needs the Stage-1 query vector instead.

Inputs (no model, no checkpoint, no transformers)
-------------------------------------------------
`<data_dir>/vocab.json`, `catalog_sid_codes.npy`, `catalog_item_ids.npy`,
`<split>/{tokens,offsets,targets}.npy`, and `item_embeddings.npy` with the
matching `item_ids.npy` (row k of the matrix is item `item_ids.npy[k]`).
A parquet with an `item_id` and an embedding column is accepted too.

The 4.78M x 256 matrix is 4.9 GB in float32; a GPU is used when available but
the CPU path works.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("embedding_knn")

DEFAULT_KS: Tuple[int, ...] = (1, 10, 100, 1000)
DEFAULT_WINDOWS: Tuple[int, ...] = (1, 3, 5)
DEFAULT_SID_DEPTHS: Tuple[int, ...] = (1, 2)
MRR_K = 10          # MRR stays at 10 even when the widest recall cut-off is 1000

# Measured on the same 98,304 validation rows against the same targets.
# The R@10 figure for popularity is exactly 1321/98304.
REFERENCE: Dict[str, Dict[str, float]] = {
    "popularity": {"R@1": 0.004395, "R@10": 1321 / 98304, "R@100": 0.047862,
                   "R@1000": 0.147288, "mrr@10": 0.006480},
    "coclick_w20": {"R@1": 0.008057, "R@10": 0.037069, "R@100": 0.109873,
                    "R@1000": 0.253021, "mrr@10": 0.015390},
    "digr": {"R@1": 0.007924, "R@10": 0.025940, "R@100": 0.075094,
             "R@1000": 0.216003, "mrr@10": 0.012580},
}


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def build_logger(level: int = logging.INFO) -> logging.Logger:
    """Console logger matching the other scripts in this directory."""
    if not logger.handlers:
        try:
            # slurm redirects stdout to a file, which Python block-buffers, so a
            # long silent load is indistinguishable from a hang
            sys.stdout.reconfigure(line_buffering=True)
        except (AttributeError, OSError):                   # pragma: no cover
            pass
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                                         datefmt="%H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def human_int(n: int) -> str:
    return f"{int(n):,}"


# --------------------------------------------------------------------------- #
# SID keys: a 5-code tuple as one sortable int64
# --------------------------------------------------------------------------- #
def pack_sid_keys(codes: np.ndarray, sizes: Sequence[int],
                  depth: Optional[int] = None) -> np.ndarray:
    """Mixed-radix pack of the first `depth` codes of each row into one int64.

    Levels are `sizes = [256, 256, 256, 256, 731]`, so the full tuple needs
    256**4 * 731 = 3.1e12, which fits int64 with room to spare. Packing instead
    of hashing keeps the lookup exact: a dict would work but costs several GB.
    """
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim == 1:
        codes = codes[None, :]
    d = codes.shape[1] if depth is None else int(depth)
    if d < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if d > codes.shape[1]:
        raise ValueError(f"depth {d} exceeds {codes.shape[1]} codes per item")
    if d > len(sizes):
        raise ValueError(f"depth {d} exceeds {len(sizes)} level sizes")
    key = np.zeros(codes.shape[0], dtype=np.int64)
    for level in range(d):
        col = codes[:, level]
        size = int(sizes[level])
        if col.size:
            lo, hi = int(col.min()), int(col.max())
            if lo < 0 or hi >= size:
                raise ValueError(
                    f"SID level {level} has codes in [{lo}, {hi}] but its "
                    f"codebook holds {size}; the vocab and the catalog disagree"
                )
        key = key * size + col
    return key


def build_prefix_index(codes: np.ndarray, sizes: Sequence[int], depth: int
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """All catalog rows sorted by prefix key, duplicates kept.

    This is the index a SID prefix needs: a depth-2 bucket holds every item that
    shares the first two codes, so the lookup must return all of them, not the
    first one. `build_key_lookup` is the other shape, one row per distinct key.
    """
    keys = pack_sid_keys(codes, sizes, depth)
    order = np.argsort(keys, kind="stable")
    return keys[order], order


def build_key_lookup(codes: np.ndarray, sizes: Sequence[int], depth: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Sorted unique prefix keys plus the catalog row that owns each one.

    A 4-code prefix can be shared by hundreds of items; the first row in catalog
    order wins, which is all we need because only the depth-5 tuple is unique.
    """
    keys = pack_sid_keys(codes, sizes, depth)
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    unique = np.ones(len(ks), dtype=bool)
    if len(ks) > 1:
        unique[1:] = ks[1:] != ks[:-1]
    return ks[unique], order[unique]


def lookup_catalog_rows(sorted_keys: np.ndarray, table_rows: np.ndarray,
                        keys: np.ndarray) -> np.ndarray:
    """Catalog row for each key, or -1 when the prefix is not in the catalog."""
    keys = np.asarray(keys, dtype=np.int64)
    pos = np.searchsorted(sorted_keys, keys)
    out = np.full(keys.shape, -1, dtype=np.int64)
    inside = pos < len(sorted_keys)
    hit = np.zeros(keys.shape, dtype=bool)
    hit[inside] = sorted_keys[pos[inside]] == keys[inside]
    out[hit] = table_rows[pos[hit]]
    return out


def bucket_slice(sorted_keys: np.ndarray, key: int) -> Tuple[int, int]:
    """Half-open range of `sorted_keys` holding one prefix."""
    lo = int(np.searchsorted(sorted_keys, key, side="left"))
    hi = int(np.searchsorted(sorted_keys, key, side="right"))
    return lo, hi


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #
def _pick_column(columns: Dict[str, np.ndarray], wanted: Iterable[str],
                 kind: str) -> str:
    lowered = {c.lower(): c for c in columns}
    for want in wanted:
        if want in lowered:
            return lowered[want]
    raise ValueError(
        f"no {kind} column among {sorted(columns)}; pass --embedding_column / "
        f"--id_column explicitly"
    )


def _column_to_2d(column) -> np.ndarray:
    """One list-of-floats parquet column as an (n, d) float32 block.

    The fast path matters: `to_numpy` on a fixed-size list either copies zero
    times or hands back an object array of 65,536 small arrays, and turning that
    second shape into a matrix is what makes a 4.9 GB read take minutes.
    """
    import pyarrow as pa

    kind = column.type
    if pa.types.is_fixed_size_list(kind):
        width = int(kind.list_size)
        flat = column.flatten().to_numpy(zero_copy_only=False)
        return np.asarray(flat, dtype=np.float32).reshape(len(column), width)
    vec = np.asarray(column.to_numpy(zero_copy_only=False))
    if vec.dtype == object:
        return np.vstack([np.asarray(v, dtype=np.float32) for v in vec])
    return np.asarray(vec, dtype=np.float32).reshape(len(vec), -1)


def _read_parquet_embeddings(path: str, id_column: Optional[str],
                             embedding_column: Optional[str]
                             ) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:                              # pragma: no cover
        raise ImportError(
            "reading a parquet needs pyarrow (`pip install pyarrow`); the numpy "
            "export `item_embeddings.npy` needs nothing"
        ) from exc

    pf = pq.ParquetFile(path)
    names = [f.name for f in pf.schema_arrow]
    id_col = id_column or _pick_column(
        {n: None for n in names}, ("item_id", "itemid", "id", "oid"), "item id")
    if embedding_column:
        emb_col = embedding_column
    else:
        lowered = {n.lower(): n for n in names}
        emb_col = None
        for want in ("embedding", "emb", "mm_emb", "vector", "vec", "features"):
            if want in lowered:
                emb_col = lowered[want]
                break
        if emb_col is None:
            emb_col = _pick_column(
                {n: None for n in names},
                tuple(n for n in names if n not in (id_col,)), "embedding")

    ids: List[np.ndarray] = []
    chunks: List[np.ndarray] = []
    total = int(pf.metadata.num_rows)
    seen = 0
    for batch in pf.iter_batches(batch_size=65_536, columns=[id_col, emb_col]):
        ids.append(np.asarray(batch.column(0)))
        chunks.append(_column_to_2d(batch.column(1)))
        seen += len(ids[-1])
        if len(ids) % 8 == 0:
            logger.info(f"  read {human_int(seen)} / {human_int(total)} "
                        f"embedding rows")
    if not ids:                                            # pragma: no cover
        raise ValueError(f"{path} holds no rows")
    logger.info(f"  read {human_int(seen)} / {human_int(total)} embedding rows")
    emb = np.concatenate(chunks, axis=0)
    return emb, np.concatenate(ids, axis=0)


def align_to_catalog(emb: np.ndarray, emb_ids: np.ndarray,
                     catalog_item_ids: np.ndarray) -> np.ndarray:
    """Reorder `emb` so row i is catalog row i.

    `ItemCatalog` stores its ids sorted ascending (`_sort_and_dedupe`), so for
    integer ids a searchsorted is the whole mapping. The catalog can also hold
    string ids (`catalog_meta.json: is_int = false`), in which case this falls
    back to a dict: slower, but it cannot be fooled by ordering. Rows whose id is
    not in the catalog are dropped, and catalog rows with no embedding stay zero.
    """
    ids = np.asarray(emb_ids)
    cat = np.asarray(catalog_item_ids)
    if len(emb) != len(ids):
        raise ValueError(f"{len(emb)} embedding rows but {len(ids)} ids")

    if ids.dtype.kind in "iu" and cat.dtype.kind in "iu":
        ids_i = ids.astype(np.int64)
        cat_i = cat.astype(np.int64)
        order = np.argsort(ids_i, kind="stable")
        pos = np.searchsorted(cat_i, ids_i[order])
        inside = pos < len(cat_i)
        hit = np.zeros(len(ids_i), dtype=bool)
        hit[inside] = cat_i[pos[inside]] == ids_i[order][inside]
        pos = np.where(hit, pos, 0)
        src = order
    else:
        lookup = {v: i for i, v in enumerate(cat.tolist())}
        pos = np.fromiter((lookup.get(v, -1) for v in ids.tolist()),
                          dtype=np.int64, count=len(ids))
        hit = pos >= 0
        src = np.arange(len(ids), dtype=np.int64)
        pos = np.where(hit, pos, 0)

    out = np.zeros((len(cat), emb.shape[1]), dtype=np.float32)
    out[pos[hit]] = emb[src[hit]]
    missing = int((~hit).sum())
    if missing:
        logger.warning(
            f"{human_int(missing)} embedding rows reference items that are not in "
            f"the catalog and were dropped"
        )
    unset = int(len(out) - hit.sum())
    if unset:
        logger.warning(f"{human_int(unset)} catalog items have no embedding row")
    return out


def load_embeddings(path: str, catalog_item_ids: np.ndarray,
                    embedding_ids: Optional[str] = None,
                    id_column: Optional[str] = None,
                    embedding_column: Optional[str] = None) -> np.ndarray:
    """Returns (n_catalog, d) float32 in catalog order."""
    lower = path.lower()
    if lower.endswith(".npy"):
        emb = np.load(path, mmap_mode="r")
        emb = np.asarray(emb, dtype=np.float32)
        logger.info(f"  loaded {path} {emb.shape}")
        if embedding_ids:
            ids = np.load(embedding_ids, allow_pickle=True)
            return align_to_catalog(emb, ids, catalog_item_ids)
        if emb.shape[0] != len(catalog_item_ids):
            raise ValueError(
                f"{path} has {emb.shape[0]} rows but the catalog has "
                f"{len(catalog_item_ids)}; pass --embedding_ids so the rows can "
                f"be aligned by item id"
            )
        logger.warning(
            "no --embedding_ids given: assuming row i of the matrix is catalog "
            "row i (true only if the export was written over the catalog order)"
        )
        return emb
    if lower.endswith(".parquet"):
        logger.info(f"  reading {path} (this is the multi-GB step)")
        emb, ids = _read_parquet_embeddings(path, id_column, embedding_column)
        logger.info(f"  aligning {human_int(len(emb))} rows to the catalog by "
                    f"item id")
        return align_to_catalog(emb, ids, catalog_item_ids)
    raise ValueError(f"unexpected embedding file {path!r}: want .npy or .parquet")


def normalise_rows(x: np.ndarray, block: int = 262_144) -> np.ndarray:
    """L2-normalise in place-friendly blocks; returns the same array."""
    for s in range(0, len(x), block):
        e = min(s + block, len(x))
        chunk = x[s:e]
        norm = np.linalg.norm(chunk, axis=1, keepdims=True)
        np.divide(chunk, np.maximum(norm, 1e-12), out=chunk)
    return x


# --------------------------------------------------------------------------- #
# row decoding
# --------------------------------------------------------------------------- #
def decode_row_windows(tokens: np.ndarray, offsets: np.ndarray, r0: int, r1: int,
                       sid_offsets: Sequence[int], num_levels: int, sep_id: int,
                       width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Decode the last `width` events of rows [r0, r1).

    Row layout (`preprocess_digr.py`, `build_tokenized_split`):

        tokens[offsets[i] + :feat_len]      feature block
        tokens[offsets[i] + feat_len]       SEP
        tokens[offsets[i] + feat_len + 1 + e*(1+L) + 0]   temporal token
        tokens[... + 1 + level]                           SID level token

    `sep_id` is 1 and every real code is >= 2 (`sid_offsets[0]` is 2, the
    temporal offset is 1757, the feature offset is 1821), so the SEP position is
    unambiguous and can be found with a single equality test.

    Returns `(codes, valid)` with `codes[r, j, level]` the code of the j-th most
    recent event (j=0 is the last one) and `valid[r, j]` false when the row has
    fewer than j+1 events.
    """
    base = int(offsets[r0])
    flat = np.asarray(tokens[base:int(offsets[r1])])
    n_rows = r1 - r0
    row_start = np.asarray(offsets[r0:r1], dtype=np.int64) - base
    row_end = np.asarray(offsets[r0 + 1:r1 + 1], dtype=np.int64) - base
    row_len = row_end - row_start

    sep_at = np.flatnonzero(flat == sep_id)
    if sep_at.size != n_rows:
        raise ValueError(
            f"expected exactly one SEP ({sep_id}) per row, found {sep_at.size} "
            f"in {n_rows} rows; the split and the vocab disagree"
        )
    feat_len = sep_at.astype(np.int64) - row_start
    if (feat_len < 0).any() or (feat_len > row_len - 1).any():
        raise ValueError("SEP position outside its row; the offsets are corrupt")
    per_event = 1 + int(num_levels)
    m = (row_len - 1 - feat_len) // per_event
    if ((m * per_event) != (row_len - 1 - feat_len)).any():
        raise ValueError(
            "row length is not 1 + feat_len + m*(1 + num_levels); the split was "
            "built with a different number of SID levels than vocab.json says"
        )

    codes = np.full((n_rows, width, int(num_levels)), -1, dtype=np.int64)
    valid = np.zeros((n_rows, width), dtype=bool)
    off = np.asarray(sid_offsets, dtype=np.int64)[: int(num_levels)]
    level_idx = 1 + np.arange(int(num_levels), dtype=np.int64)
    for j in range(width):
        start = row_end - (j + 1) * per_event
        ok = m > j
        valid[:, j] = ok
        if not ok.any():
            continue
        idx = start[:, None] + level_idx[None, :]
        gathered = flat[np.clip(idx, 0, flat.size - 1)] - off[None, :]
        codes[:, j, :] = np.where(ok[:, None], gathered, -1)
    return codes, valid


# --------------------------------------------------------------------------- #
# top-k over a candidate matrix
# --------------------------------------------------------------------------- #
def topk_over(queries, candidates, topk: int, chunk: int = 262_144):
    """Exact top-k of `queries @ candidates.T`, memory bounded by `chunk`.

    Both arguments are torch tensors on the same device. A single 512 x 4.78M
    score matrix is 9.8 GB, so the catalog is scanned in blocks with a running
    top-k merge, the same trick `DiGRRetriever.item_logits` uses.
    """
    import torch

    n_cand = int(candidates.shape[0])
    k = max(1, min(int(topk), n_cand))
    best_v = None
    best_i = None
    for s in range(0, n_cand, chunk):
        e = min(s + chunk, n_cand)
        scores = queries @ candidates[s:e].T
        if best_v is None:
            # the first block may be narrower than k when the candidate matrix
            # itself is small (a SID bucket, or a smoke test)
            best_v, best_i = torch.topk(scores, min(k, e - s), dim=1)
            if k < n_cand:
                best_i = best_i + s
            continue
        cand_i = torch.arange(s, e, device=scores.device, dtype=torch.int64)
        cand_i = cand_i.expand(scores.shape[0], -1)
        merged = torch.cat([best_v, scores], dim=1)
        best_v, pick = merged.topk(min(k, merged.shape[1]), dim=1)
        best_i = torch.cat([best_i, cand_i], dim=1).gather(1, pick)
    return best_v, best_i


def ranks_from_topk(indices, targets, topk: int):
    """1-based rank of the target, or `topk + 1` when it is not retrieved.

    The miss rank is `topk + 1` and never 0: a 0 would pass every `rank <= k`
    test and make `1 / log2(rank + 1)` a division by zero. Same convention as
    `train_digr.evaluate` and `baselines_digr.accumulate`.
    """
    import torch

    match = indices == targets[:, None]
    found = match.any(dim=1)
    pos = match.to(torch.int64).argmax(dim=1) + 1
    return torch.where(found, pos, torch.full_like(pos, int(topk) + 1))


def new_metrics(ks: Sequence[int]) -> Dict[str, float]:
    out: Dict[str, float] = {"n": 0.0, "mrr": 0.0}
    for k in ks:
        out[f"recall@{k}"] = 0.0
        out[f"ndcg@{k}"] = 0.0
    return out


def update_metrics(metrics: Dict[str, float], ranks: np.ndarray,
                   ks: Sequence[int]) -> None:
    ranks = np.asarray(ranks, dtype=np.int64)
    metrics["n"] += float(ranks.size)
    for k in ks:
        hit = ranks <= k
        metrics[f"recall@{int(k)}"] += float(hit.sum())
        metrics[f"ndcg@{int(k)}"] += float(
            (1.0 / np.log2(ranks[hit] + 1.0)).sum()
        )
    metrics["mrr"] += float((1.0 / ranks[ranks <= MRR_K]).sum())


def finalise_metrics(metrics: Dict[str, float], ks: Sequence[int]
                     ) -> Dict[str, float]:
    n = max(metrics.get("n", 0.0), 1.0)
    out = {"n": metrics["n"]}
    for k in ks:
        out[f"recall@{int(k)}"] = metrics[f"recall@{int(k)}"] / n
        out[f"ndcg@{int(k)}"] = metrics[f"ndcg@{int(k)}"] / n
    out["mrr"] = metrics["mrr"] / n
    return out


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def verdict(r10: float) -> str:
    if r10 >= 0.05:
        return ("embedding knows the next click; the SID is destroying it. "
                "Serve FAISS + reranker and redesign TIGER before writing it")
    if r10 >= 0.025:
        return ("embedding, co-click and DiGR all know the same amount; the SID "
                "is exonerated and TIGER proceeds as specified")
    if r10 < 0.02:
        return ("item geometry does not encode intent; behaviour lives in the "
                "sequence tower and the last-item query is the wrong probe")
    return "between the two bands: read the depth-1 and depth-2 rows before deciding"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_cli(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace,
                                                            argparse.Namespace]:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", required=True,
                   help="preprocessed DiGR data dir (vocab.json, catalog_*.npy, "
                        "<split>/{tokens,offsets,targets}.npy)")
    p.add_argument("--embeddings", required=True,
                   help="item_embeddings.npy (or .parquet) from the Stage-1 export")
    p.add_argument("--embedding_ids", default=None,
                   help="item_ids.npy, the row-order ids of --embeddings. Omit "
                        "only if the matrix is already in catalog order")
    p.add_argument("--id_column", default=None,
                   help="parquet only: item id column name")
    p.add_argument("--embedding_column", default=None,
                   help="parquet only: embedding column name")
    p.add_argument("--split", default="val",
                   help="split to score (default val, the split everything else "
                        "reports on)")
    p.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS),
                   help="recall cut-offs (default 1,10,100,1000)")
    p.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS),
                   help="history windows for the kNN query (default 1,3,5)")
    p.add_argument("--sid_depths", default=",".join(str(d) for d in DEFAULT_SID_DEPTHS),
                   help="SID prefix depths to test as an index (default 1,2). "
                        "Pass an empty string to skip")
    p.add_argument("--topk", type=int, default=1000,
                   help="how deep to retrieve per row; must be >= max(ks)")
    p.add_argument("--max_rows", type=int, default=0,
                   help="score only the first N rows (smoke test)")
    p.add_argument("--row_chunk", type=int, default=2048,
                   help="rows decoded per step")
    p.add_argument("--query_chunk", type=int, default=256,
                   help="queries scored per step; the score block is "
                        "query_chunk x catalog_chunk")
    p.add_argument("--catalog_chunk", type=int, default=262_144,
                   help="catalog rows per score block")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    p.add_argument("--skip_train_stats", action="store_true",
                   help="skip the train-target frequency diagnostic")
    p.add_argument("--out_json", default=None)
    args, _ = p.parse_known_args(argv)

    cli = argparse.Namespace(
        ks=tuple(sorted({int(x) for x in args.ks.split(",") if x.strip()})),
        windows=tuple(sorted({int(x) for x in args.windows.split(",") if x.strip()})),
        sid_depths=tuple(sorted({int(x) for x in args.sid_depths.split(",")
                                 if x.strip()})) if args.sid_depths.strip() else (),
    )
    if not cli.ks or cli.ks[0] < 1:
        p.error("--ks must be positive integers")
    if not cli.windows or cli.windows[0] < 1:
        p.error("--windows must be positive integers")
    if any(d < 1 for d in cli.sid_depths):
        p.error("--sid_depths must be positive integers")
    if args.topk < max(cli.ks):
        p.error(f"--topk {args.topk} is smaller than the widest cut-off "
                f"{max(cli.ks)}; recall above the retrieval depth is meaningless")
    if args.row_chunk < 1 or args.query_chunk < 1 or args.catalog_chunk < 1:
        p.error("chunk sizes must be >= 1")
    return args, cli


def resolve_device(request: str):
    import torch

    if request == "cpu":
        return torch.device("cpu")
    if request == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    args, cli = parse_cli(argv)
    log = build_logger()
    import torch

    device = resolve_device(args.device)
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    t_start = time.time()
    log.info(f"data_dir   {args.data_dir}")
    log.info(f"embeddings {args.embeddings}")

    # ---- vocab + catalog -------------------------------------------------- #
    vocab_path = os.path.join(args.data_dir, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    sid_sizes = [int(s) for s in vocab["sid_sizes"]]
    sid_offsets = [int(o) for o in vocab["sid_offsets"]]
    num_levels = int(vocab.get("num_levels", len(sid_sizes)))
    sep_id = int(vocab.get("sep_id", 1))
    if len(sid_offsets) != num_levels:
        raise ValueError("vocab.json sid_offsets and num_levels disagree")
    log.info(f"vocab      {num_levels} SID levels, sizes {sid_sizes}")

    cat_codes = np.load(os.path.join(args.data_dir, "catalog_sid_codes.npy"))
    cat_ids = np.load(os.path.join(args.data_dir, "catalog_item_ids.npy"),
                      allow_pickle=True)
    n_catalog = len(cat_codes)
    if len(cat_ids) != n_catalog:
        raise ValueError("catalog_item_ids.npy and catalog_sid_codes.npy disagree")
    log.info(f"catalog    {human_int(n_catalog)} items, {cat_codes.dtype} codes")

    # ---- split ------------------------------------------------------------ #
    split_dir = os.path.join(args.data_dir, args.split)
    log.info(f"split      reading {split_dir} ...")
    tokens = np.load(os.path.join(split_dir, "tokens.npy"), mmap_mode="r")
    offsets = np.load(os.path.join(split_dir, "offsets.npy"))
    targets = np.load(os.path.join(split_dir, "targets.npy"))
    n_rows = len(targets)
    if args.max_rows:
        n_rows = min(n_rows, int(args.max_rows))
    if len(offsets) != len(targets) + 1:
        raise ValueError("offsets.npy and targets.npy disagree on the row count")
    log.info(f"split      {human_int(n_rows)} rows, {human_int(len(tokens))} tokens")

    # ---- embeddings ------------------------------------------------------- #
    emb = load_embeddings(args.embeddings, cat_ids, args.embedding_ids,
                          args.id_column, args.embedding_column)
    if emb.shape[0] != n_catalog:
        raise ValueError(f"embeddings have {emb.shape[0]} rows, catalog "
                         f"{n_catalog}")
    norms = np.linalg.norm(emb[:min(len(emb), 50_000)], axis=1)
    if np.abs(norms - 1.0).max() > 1e-3:
        log.warning(f"embedding rows are not unit norm (mean {norms.mean():.4f}); "
                    f"renormalising")
        emb = normalise_rows(emb)
    else:
        log.info(f"embedding rows are unit norm (mean {norms.mean():.6f})")
    dim = emb.shape[1]
    log.info(f"matrix     {emb.shape[0]} x {dim} float32 = "
             f"{emb.nbytes / 1e9:.2f} GB on host; moving to {device} ...")
    catalog_t = torch.from_numpy(emb).to(device)
    log.info(f"matrix     resident on {device}")

    log.info("=" * 78)
    log.info(f"  split        : {args.split}  ({human_int(n_rows)} rows of "
             f"{human_int(len(targets))})")
    log.info(f"  catalog      : {human_int(n_catalog)} items x {dim}-d")
    log.info(f"  device/dtype : {device} / {args.dtype}")
    log.info(f"  windows      : {list(cli.windows)}   cut-offs {list(cli.ks)}   "
             f"topk {args.topk}")
    log.info(f"  sid depths   : {list(cli.sid_depths) or 'skipped'}")
    log.info("=" * 78)

    # ---- prefix tables ---------------------------------------------------- #
    depth_tables: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for depth in cli.sid_depths:
        if depth > num_levels:
            raise ValueError(f"--sid_depths {depth} but the catalog has "
                             f"{num_levels} levels")
        depth_tables[depth] = build_prefix_index(cat_codes, sid_sizes, depth)
    full_keys, full_rows = build_key_lookup(cat_codes, sid_sizes, num_levels)
    log.info(f"  SID key table: {human_int(len(full_keys))} distinct "
             f"{num_levels}-tuples for {human_int(n_catalog)} items")

    # ---- optional popularity diagnostic ---------------------------------- #
    ever_target = None
    base_rate = math.nan
    if not args.skip_train_stats:
        train_targets = os.path.join(args.data_dir, "train", "targets.npy")
        if os.path.exists(train_targets):
            tt = np.load(train_targets)
            ever_target = np.bincount(tt, minlength=n_catalog) > 0
            base_rate = float(ever_target.mean())
            log.info(f"  popularity   : {human_int(int(ever_target.sum()))} of "
                     f"{human_int(n_catalog)} items are a train target "
                     f"({100 * base_rate:.2f}% base rate for top-10 hits)")
        else:
            log.warning(f"{train_targets} missing; skipping the popularity check")

    # ---- probes ----------------------------------------------------------- #
    width = max(cli.windows)
    knn = {w: new_metrics(cli.ks) for w in cli.windows}
    diag = {
        "repeat_rate": 0.0,
        "last_item_unmapped": 0.0,
        "rows": 0.0,
    }
    top10_in_train: Dict[int, float] = {w: 0.0 for w in cli.windows}

    q_last = np.zeros((n_rows, dim), dtype=np.float32)
    cat_last = np.full(n_rows, -1, dtype=np.int64)
    last_codes = np.full((n_rows, num_levels), -1, dtype=np.int64)

    log.info("scoring kNN probes ...")
    for r0 in range(0, n_rows, args.row_chunk):
        r1 = min(r0 + args.row_chunk, n_rows)
        codes, valid = decode_row_windows(
            tokens, offsets, r0, r1, sid_offsets, num_levels, sep_id, width)
        n = r1 - r0
        # Invalid slots hold -1; pack zeros in their place and let `valid` mask
        # them out, so a short history cannot trip the codebook range check.
        safe_codes = np.where(valid[:, :, None], codes, 0)
        row_keys = pack_sid_keys(
            safe_codes.reshape(-1, num_levels), sid_sizes).reshape(n, width)
        cat_of_event = np.where(
            valid.reshape(-1),
            lookup_catalog_rows(full_keys, full_rows, row_keys.reshape(-1)),
            -1,
        ).reshape(n, width)
        tgt = torch.from_numpy(np.asarray(targets[r0:r1], dtype=np.int64)).to(device)
        tgt_np = np.asarray(targets[r0:r1], dtype=np.int64)

        last_codes[r0:r1] = codes[:, 0, :]
        cat_last[r0:r1] = cat_of_event[:, 0]
        diag["rows"] += float(n)
        diag["last_item_unmapped"] += float((cat_of_event[:, 0] < 0).sum())
        diag["repeat_rate"] += float(
            ((cat_of_event == tgt_np[:, None]) & valid).any(axis=1).sum())

        gather = np.where(valid, cat_of_event, 0)
        vecs = emb[gather]                                   # (n, width, dim)
        vecs_t = torch.from_numpy(vecs).to(device, dtype=amp_dtype)

        for w in cli.windows:
            use = (np.arange(width) < min(w, width))[None, :] & valid
            count = np.maximum(use.sum(axis=1), 1)
            summed = (vecs * use[:, :, None]).sum(axis=1) / count[:, None]
            q = torch.from_numpy(np.asarray(summed, dtype=np.float32)).to(
                device, dtype=amp_dtype)
            q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
            if w == 1:
                q_last[r0:r1] = q.float().cpu().numpy()

            for s in range(0, n, args.query_chunk):
                e = min(s + args.query_chunk, n)
                _, idx = topk_over(q[s:e], catalog_t, args.topk,
                                   args.catalog_chunk)
                ranks = ranks_from_topk(idx, tgt[s:e], args.topk)
                update_metrics(knn[w], ranks.cpu().numpy(), cli.ks)
                if ever_target is not None:
                    top10 = idx[:, :min(10, idx.shape[1])].cpu().numpy()
                    top10_in_train[w] += float(
                        ever_target[top10].sum()) / max(top10.shape[1], 1)

    # ---- SID prefix probes ------------------------------------------------ #
    sid_metrics: Dict[int, Dict[str, float]] = {}
    sid_cand: Dict[int, float] = {}
    sid_bucket_mean: Dict[int, float] = {}
    if cli.sid_depths:
        log.info("scoring SID prefix probes ...")
        for depth in cli.sid_depths:
            keys_sorted, rows_sorted = depth_tables[depth]
            q_keys = pack_sid_keys(last_codes, sid_sizes, depth)
            unique_keys, inverse = np.unique(q_keys, return_inverse=True)
            metrics = new_metrics(cli.ks)
            hits = 0.0
            cand_total = 0.0
            q_last_t = torch.from_numpy(q_last).to(device, dtype=amp_dtype)
            for k_i, key in enumerate(unique_keys):
                rows = np.flatnonzero(inverse == k_i)
                if not rows.size:
                    continue
                lo, hi = bucket_slice(keys_sorted, int(key))
                bucket = rows_sorted[lo:hi]
                cand_total += float(bucket.size) * rows.size
                tgt_np = np.asarray(targets[:n_rows][rows], dtype=np.int64)
                bucket_lookup = np.sort(bucket)
                pos = np.searchsorted(bucket_lookup, tgt_np)
                inside = pos < bucket_lookup.size
                if inside.any():
                    hits += float((bucket_lookup[pos[inside]] == tgt_np[inside]).sum())
                if not bucket.size:
                    continue
                cand_t = catalog_t[torch.from_numpy(bucket).to(device)]
                tgt = torch.from_numpy(tgt_np).to(device)
                for s in range(0, rows.size, args.query_chunk):
                    e = min(s + args.query_chunk, rows.size)
                    sel = torch.from_numpy(rows[s:e].astype(np.int64)).to(device)
                    _, local = topk_over(q_last_t[sel], cand_t, args.topk,
                                         args.catalog_chunk)
                    mapped = torch.from_numpy(bucket).to(device)[local]
                    ranks = ranks_from_topk(mapped, tgt[s:e], args.topk)
                    update_metrics(metrics, ranks.cpu().numpy(), cli.ks)
            sid_metrics[depth] = finalise_metrics(metrics, cli.ks)
            sid_cand[depth] = hits / max(float(n_rows), 1.0)
            sid_bucket_mean[depth] = cand_total / max(float(n_rows), 1.0)

    # ---- report ----------------------------------------------------------- #
    def line(name: str, m: Dict[str, float]) -> str:
        cells = [f"{name:<18}"]
        for k in cli.ks:
            value = m.get(f"recall@{k}", math.nan)
            shown = "      -" if not np.isfinite(value) else f"{100 * value:7.4f}%"
            cells.append(f"R@{k} {shown}")
        cells.append(f"MRR@{MRR_K} {m.get('mrr', math.nan):.5f}")
        return "  ".join(cells)

    log.info("-" * 78)
    log.info(f"  {'reference (same 98,304 rows, same targets)':<18}")
    for name in ("popularity", "coclick_w20", "digr"):
        ref = REFERENCE[name]
        m = {f"recall@{k}": ref.get(f"R@{k}", math.nan) for k in cli.ks}
        m["mrr"] = ref["mrr@10"]
        log.info(line(name, m))

    log.info("-" * 78)
    log.info(f"  {'embedding kNN':<18}")
    knn_final: Dict[int, Dict[str, float]] = {}
    for w in cli.windows:
        knn_final[w] = finalise_metrics(knn[w], cli.ks)
        log.info(line(f"emb kNN w{w}", knn_final[w]))

    if cli.sid_depths:
        log.info("-" * 78)
        log.info(f"  {'SID prefix + cosine rerank':<18}")
        for depth in cli.sid_depths:
            log.info(line(f"sid{depth} + cosine", sid_metrics[depth]))
            log.info(f"    depth {depth}: candidate recall "
                     f"{100 * sid_cand[depth]:7.4f}%   mean bucket "
                     f"{sid_bucket_mean[depth]:10.1f} items")

    rows = max(diag["rows"], 1.0)
    log.info("-" * 78)
    log.info(f"  target already in the row's own history: "
             f"{100 * diag['repeat_rate'] / rows:.2f}%")
    log.info(f"  last clicked item not found in the catalog: "
             f"{100 * diag['last_item_unmapped'] / rows:.4f}%")
    if ever_target is not None:
        for w in cli.windows:
            share = top10_in_train[w] / max(rows, 1.0)
            log.info(f"  emb kNN w{w}: {100 * share:.2f}% of top-10 slots are "
                     f"train targets vs {100 * base_rate:.2f}% base rate "
                     f"({share / max(base_rate, 1e-9):.2f}x)")
    # The verdict is defined on R@10; fall back to the widest cut-off asked for
    # so a smoke test with --ks 1,3 still says something instead of reading a
    # missing key as 0% and declaring the embedding useless.
    read_k = 10 if 10 in cli.ks else max(cli.ks)
    read_key = f"recall@{read_k}"
    best_w = max(cli.windows, key=lambda w: knn_final[w].get(read_key, -1.0))
    headline = knn_final[best_w].get(read_key, 0.0)
    note = "" if read_k == 10 else f"  [read on R@{read_k}, 10 not in --ks]"
    log.info("-" * 78)
    log.info(f"  VERDICT (best window w{best_w}, R@{read_k} "
             f"{100 * headline:.4f}%): {verdict(headline)}{note}")
    log.info(f"  elapsed {time.time() - t_start:.0f}s")
    log.info("=" * 78)

    if args.out_json:
        payload = {
            "split": args.split,
            "rows": int(n_rows),
            "catalog_size": int(n_catalog),
            "embedding_dim": int(dim),
            "embeddings": args.embeddings,
            "ks": list(cli.ks),
            "topk": int(args.topk),
            "mrr_k": MRR_K,
            "reference": REFERENCE,
            "knn": {f"w{w}": knn_final[w] for w in cli.windows},
            "sid_prefix": {
                f"depth{d}": dict(sid_metrics[d],
                                  candidate_recall=sid_cand[d],
                                  mean_bucket=sid_bucket_mean[d])
                for d in cli.sid_depths
            },
            "repeat_rate": diag["repeat_rate"] / rows,
            "last_item_unmapped": diag["last_item_unmapped"] / rows,
            "top10_train_target_share": {
                f"w{w}": top10_in_train[w] / rows for w in cli.windows
            } if ever_target is not None else {},
            "verdict": verdict(headline),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        log.info(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":                      # pragma: no cover
    raise SystemExit(main())
