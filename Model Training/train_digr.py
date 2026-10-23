#!/usr/bin/env python3
"""
DiGR: LLM-backbone generative retrieval from RQ-VAE Semantic IDs
=============================================================================
Implements the DiGR plan:

  Phase 1 - vocabulary partition + catalog SID buffer
  Phase 2 - Qwen backbone with LoRA (attention projections) and trainable
            input embeddings (`modules_to_save=["embed_tokens"]`)
  Phase 3 - dual-branch forward pass:
              h_x = backbone(sequence)[last non-pad token]
              V   = mean_l E[catalog_sid_token_l]          (item branch)
              z   = h_x V^T / tau
              loss = full-catalog cross entropy vs target catalog index
  Phase 4 - freeze, precompute V once, retrieve top-K by dot product
            (torch matmul, chunked; the same vectors can be dumped for FAISS)

--------------------------------------------------------------------------------
INPUTS
--------------------------------------------------------------------------------
This file consumes a *preprocessed* dataset directory (`--data_dir`, default
`digr_data/`) produced by `preprocess_digr.py`.  Raw parquet parsing, catalog
loading, user features, gap bins and tokenisation all live in that phase; this
file only trains / infers over the written CSR arrays.

  digr_data/
    preprocess_meta.json      manifest (sizes, schema, hyper-parameters)
    vocab.json                the token layout (offsets per level)
    catalog_item_ids.npy      |I| raw item ids      (target = row index)
    catalog_sid_codes.npy     |I| x L SID codes
    catalog_meta.json         num_levels / codebook sizes
    users.npy, users_meta.json  legacy empty profile vocabulary (unused)
    user_feat_meta.json       static feature columns (M tokens per row)
    gap_edges.npy             temporal bin edges
    train/, val/ (optional infer/)   tokens.npy offsets.npy targets.npy
                                      users.npy stats.json

Every directory is validated before a run starts; a missing/stale one fails fast
with the exact `preprocess_digr.py` command to re-run.

--------------------------------------------------------------------------------
TOKEN LAYOUT (one slot per interaction, exactly as in the plan)
--------------------------------------------------------------------------------
  [F_1 .. F_M, SEP, (T_1, S_1_1 .. S_1_L), ...]   len = M + 1 + T*(1+L)

  F_j = static user-feature token = offset_features + (slot j of the user's
        feature block); M = 0 when no --user_feat_path was given, so a row
        without static context is just [SEP, (T, S..)*].  The block is a
        property of the user, not of the event, and it never enters the item
        branch.
  T_i = temporal token = discretised gap to the previous interaction, log-spaced
        bins over [1s, --max_gap_seconds]; the first interaction is bin 0.
        (Hour-of-day / day-of-week tokens would need a second temporal slot; the
        plan's layout has one, so only the gap is encoded.  Widen
        --n_temporal_bins for finer gap resolution.)
  S_i = SID Level l token = Offset_SID_l + code_l, l = 1..L

The plan's `V` uses only the SID tokens; U and T never enter the item branch.

--------------------------------------------------------------------------------
H800 PERFORMANCE NOTES
--------------------------------------------------------------------------------
* Training never materialises the [|I|, d] catalog matrix.  Because
  V = mean_l E[catalog_token_l], we use the exactly equivalent factorisation
      z_i = (1/L) * sum_l h_x . E[catalog_token_{i,l}] / tau
          = (1/L) * sum_l g[catalog_token_{i,l}] / tau,  g = h_x E^T
  where E is the (tiny, ~10k rows) token embedding table.  Gradients reach the
  SID embedding rows exactly as in the V formulation, at a fraction of the
  memory.  `torch.cat` over catalog chunks keeps it explicit and simple.
* bf16 + FlashAttention-2 + gradient checkpointing + fused AdamW.
* Tokenisation is not repeated here: the int32 CSR buffers are loaded once from
  `digr_data/<split>/`; collation is a bulk numpy pad, and a length-bucketed
  batch sampler minimises padding.
* Inference precomputes V once and does chunked top-K, so the [B, |I|] logits
  are never fully resident (and V can be exported for FAISS).

Only this file is modified; the data phase lives in `preprocess_digr.py`.

--------------------------------------------------------------------------------
QUICK START
--------------------------------------------------------------------------------
  # 0) one-off preprocessing -> digr_data/  (see preprocess_digr.py --help)
  python preprocess_digr.py --sequence_path data/seq.parquet \
      --catalog_path data/index_rqvae_prod.json --output_dir digr_data

  # 1) train  ->  digr_runs/digr_<timestamp>/
  python train_digr.py --mode train --data_dir digr_data \
      --batch_size 32 --grad_accum 2 --epochs 3 --compile

  # 2) top-K retrieval with the frozen catalog matrix V
  python train_digr.py --mode infer \
      --checkpoint digr_runs/digr_<timestamp>/final \
      --data_dir digr_data --infer_split val --topk 20

  # 3) dump V (|I| x d) for FAISS / a vector database
  python train_digr.py --mode export \
      --checkpoint digr_runs/digr_<timestamp>/final

Artefacts written per run directory:
  adapter_model.safetensors + adapter_config.json   (LoRA + modules_to_save)
  vocab.json, catalog_item_ids.npy, catalog_sid_codes.npy, catalog_meta.json
  users.npy, users_meta.json, gap_edges.npy, head_config.json
  train_state.pt (epoch/step/best/optimizer/scheduler/scaler)
  catalog_vectors.npy  (|I| x d, the Phase-4 inference cache)
  best/  final/  logs/train.log  (infer writes predictions.parquet)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyarrow is required to write predictions.parquet.") from exc

# The preprocessing phase owns the token layout; import the shared data types
# (and the on-disk split format) from it instead of duplicating them.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    from preprocess_digr import (  # noqa: E402  (path fix above)
        MANIFEST_NAME,
        ItemCatalog,
        TokenizedSplit,
        UserMap,
        Vocab,
        load_split,
        split_exists,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "preprocess_digr.py must live next to train_digr.py (shared data types)."
    ) from exc

try:
    from transformers import AutoModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit("transformers is required (AutoModel).") from exc

def _require_peft():
    """Import peft lazily so `--help` and pure-data modes still work without it."""
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "peft is required for LoRA training/inference: pip install 'peft>=0.14'"
        ) from exc
    return LoraConfig, PeftModel, get_peft_model


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SEED = 42
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B"
DEFAULT_DATA_DIR = "digr_data"
DEFAULT_OUTPUT_DIR = "digr_runs"

DEFAULT_BATCH_SIZE = 64
DEFAULT_EVAL_BATCH_SIZE = 128

DEFAULT_LR = 2e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_EMBED_LR_MULT = 2.0
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_MIN_LR_RATIO = 0.1
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_LABEL_SMOOTHING = 0.0
DEFAULT_TEMPERATURE = 0.07

DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LOGIT_CHUNK = 500_000

BANNER_WIDTH = 88


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("DiGR")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
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




# ---------------------------------------------------------------------------
# Datasets, batch sampler, collation
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    def __init__(self, split: TokenizedSplit, indices: np.ndarray):
        self.tokens = split.tokens
        self.offsets = split.offsets
        self.targets = split.targets
        self.indices = np.asarray(indices, dtype=np.int64)
        self.lengths = split.lengths[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        s = int(self.offsets[j])
        e = int(self.offsets[j + 1])
        return self.tokens[s:e], int(self.targets[j]), j


def make_collate(pad_id: int):
    """Top-level picklable collate (a closure cannot be sent to spawn workers)."""
    return PadCollate(pad_id)


class PadCollate:
    """Bulk right-pad a batch of variable-length token sequences."""

    def __init__(self, pad_id: int):
        self.pad_id = int(pad_id)

    def __call__(self, batch):
        b = len(batch)
        max_len = max(len(item[0]) for item in batch)
        ids = np.full((b, max_len), self.pad_id, dtype=np.int64)
        mask = np.zeros((b, max_len), dtype=np.int64)
        targets = np.empty(b, dtype=np.int64)
        rows = np.empty(b, dtype=np.int64)
        for i, (tok, y, j) in enumerate(batch):
            ln = len(tok)
            ids[i, :ln] = tok
            mask[i, :ln] = 1
            targets[i] = y
            rows[i] = j
        return (
            torch.from_numpy(ids),
            torch.from_numpy(mask),
            torch.from_numpy(targets),
            torch.from_numpy(rows),
        )


class LengthBucketBatchSampler(Sampler):
    """Sort by length, shuffle inside pools, shuffle batch order (less padding)."""

    def __init__(self, lengths: np.ndarray, batch_size: int, shuffle: bool = True,
                 seed: int = SEED, bucket_factor: int = 50, drop_last: bool = True):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.pool_size = max(self.batch_size, self.batch_size * int(bucket_factor))
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.drop_last:
            self._len = len(self.lengths) // self.batch_size
        else:
            self._len = math.ceil(len(self.lengths) / self.batch_size)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = np.argsort(self.lengths, kind="stable")

        if self.shuffle:
            pools = []
            for start in range(0, len(order), self.pool_size):
                pool = order[start:start + self.pool_size].copy()
                rng.shuffle(pool)
                pools.append(pool)
            order = np.concatenate(pools) if pools else order

        batches = [
            order[i:i + self.batch_size]
            for i in range(0, len(order), self.batch_size)
        ]
        if self.drop_last:
            batches = [b for b in batches if len(b) == self.batch_size]
        if self.shuffle:
            rng.shuffle(batches)

        for b in batches:
            yield b.tolist()


# ---------------------------------------------------------------------------
# Phase 2/3: backbone + retrieval head
# ---------------------------------------------------------------------------

def _embedding_layer(backbone: torch.nn.Module) -> torch.nn.Module:
    m = backbone
    if hasattr(m, "backbone"):  # DiGRRetriever wrapper
        m = m.backbone
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    return m.get_input_embeddings()


def _embedding_weight(backbone: torch.nn.Module) -> torch.Tensor:
    layer = _embedding_layer(backbone)
    # PEFT `modules_to_save` wraps the layer; the trainable copy lives inside.
    if hasattr(layer, "modules_to_save"):
        for sub in layer.modules_to_save.values():
            return sub.weight
    if hasattr(layer, "original_module"):
        return layer.original_module.weight
    return layer.weight


class DiGRRetriever(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, catalog_token_ids: np.ndarray,
                 temperature: float, l2_normalize: bool, logit_chunk: int):
        super().__init__()
        self.backbone = backbone
        toks = torch.as_tensor(np.asarray(catalog_token_ids, dtype=np.int64), dtype=torch.long)
        self.register_buffer("catalog_token_ids", toks, persistent=False)
        self.temperature = float(temperature)
        self.l2_normalize = bool(l2_normalize)
        self.logit_chunk = int(logit_chunk)
        self.num_levels = int(toks.shape[1])

    @property
    def catalog_size(self) -> int:
        return int(self.catalog_token_ids.shape[0])

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state
        last_idx = attention_mask.sum(dim=1).clamp(min=1) - 1
        b = hidden.shape[0]
        h = hidden[torch.arange(b, device=hidden.device), last_idx]
        return h

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encode(input_ids, attention_mask)

    def item_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Full-catalog logits via the exact V=E[catalog_sids] factorisation."""
        w = _embedding_weight(self.backbone)
        if self.l2_normalize:
            h = F.normalize(h, dim=-1)
            w = F.normalize(w.float(), dim=-1).to(w.dtype)

        # [B, vocab] projection; vocab is tiny (~10k), the catalog is not.
        # fp32 accumulation of the SID sum: bf16 logits would add ~1e-2 noise on
        # the CE input at tau=0.07, for a [B, chunk] tensor only.
        g = torch.matmul(h, w.t()).float()
        scale = 1.0 / (self.num_levels * self.temperature)

        chunks: List[torch.Tensor] = []
        n = self.catalog_size
        for start in range(0, n, self.logit_chunk):
            toks = self.catalog_token_ids[start:start + self.logit_chunk]
            acc = g.index_select(1, toks[:, 0])
            for level in range(1, self.num_levels):
                acc = acc + g.index_select(1, toks[:, level])
            chunks.append(acc * scale)
        return torch.cat(chunks, dim=1)

    @torch.inference_mode()
    def precompute_vectors(self, dtype=torch.bfloat16) -> torch.Tensor:
        """V = (1/L) sum_l E[catalog_sid_l] in R^{|I| x d} (Phase 4).

        Streamed per level and per chunk so the [|I|, d] matrix is only ever
        allocated once, in the output dtype (no fp32 duplicate).
        """
        w = _embedding_weight(self.backbone)
        toks = self.catalog_token_ids
        d = int(w.shape[1])
        v = torch.zeros(self.catalog_size, d, dtype=dtype, device=w.device)
        step = max(1, min(int(self.logit_chunk), 65536))
        for level in range(self.num_levels):
            idx = toks[:, level]
            for start in range(0, self.catalog_size, step):
                block = w.index_select(0, idx[start:start + step]).float()
                if self.l2_normalize:
                    block = F.normalize(block, dim=-1)
                v[start:start + step] += block.to(dtype)
        v /= float(self.num_levels)
        return v

    @torch.inference_mode()
    def retrieve(self, h: torch.Tensor, vectors: torch.Tensor, topk: int,
                 chunk: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chunked top-K over the full catalog (bounded [B, chunk] logits).

        Returns (catalog_indices, scores), both [B, topk].
        """
        if self.l2_normalize:
            h = F.normalize(h.float(), dim=-1).to(h.dtype)
        n = vectors.shape[0]
        v_dtype = h.dtype
        cand_scores: List[torch.Tensor] = []
        cand_idx: List[torch.Tensor] = []
        for start in range(0, n, chunk):
            block = vectors[start:start + chunk].to(v_dtype)
            scores = torch.matmul(h, block.t()) / self.temperature
            k = min(int(topk), scores.shape[1])
            top = torch.topk(scores, k, dim=1)
            cand_scores.append(top.values)
            cand_idx.append(top.indices + start)
        scores = torch.cat(cand_scores, dim=1)
        idx = torch.cat(cand_idx, dim=1)
        k = min(int(topk), scores.shape[1])
        top = torch.topk(scores, k, dim=1)
        return torch.gather(idx, 1, top.indices), top.values


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def load_backbone(model_name: str, vocab_size: int, args, logger: logging.Logger):
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    load_dtype = dtype_map[args.dtype]

    impls = [args.attn_implementation] if args.attn_implementation else [
        "flash_attention_2", "sdpa", "eager"
    ]

    backbone = None
    last_err: Optional[Exception] = None
    for impl in impls:
        kwargs = dict(
            attn_implementation=impl,
            trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
            local_files_only=bool(args.local_files_only),
        )
        try:
            try:
                backbone = AutoModel.from_pretrained(model_name, dtype=load_dtype, **kwargs)
            except TypeError:
                backbone = AutoModel.from_pretrained(model_name, torch_dtype=load_dtype, **kwargs)
            logger.info(f"  + backbone loaded ({model_name}, attn={impl}, dtype={args.dtype})")
            break
        except Exception as exc:  # noqa: BLE001 - fall back to the next attention backend
            last_err = exc
            logger.warning(f"  ! could not load with attn={impl}: {exc}")
            backbone = None

    if backbone is None:
        raise RuntimeError(f"failed to load backbone {model_name}: {last_err}")

    try:
        backbone.resize_token_embeddings(int(vocab_size), mean_resizing=False)
    except TypeError:
        backbone.resize_token_embeddings(int(vocab_size))

    if args.reinit_embeddings:
        with torch.no_grad():
            emb = backbone.get_input_embeddings()
            emb.weight.normal_(mean=0.0, std=0.02)
        logger.info("  + input embeddings re-initialised (fresh SID/temporal/feature rows)")

    backbone.config.vocab_size = int(vocab_size)
    backbone.config.pad_token_id = 0
    backbone.config.eos_token_id = 1
    backbone.config.bos_token_id = 1
    backbone.config.use_cache = False
    backbone.config.tie_word_embeddings = False
    return backbone


def _enable_gradient_checkpointing(model: torch.nn.Module, logger: logging.Logger) -> None:
    targets = [model]
    base = getattr(model, "base_model", None)
    if base is not None and hasattr(base, "model"):
        targets.append(base.model)
    last: Optional[Exception] = None
    for target in targets:
        try:
            target.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info("  + gradient checkpointing enabled")
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
    logger.warning(f"  ! gradient checkpointing could not be enabled: {last}")


def build_peft_backbone(backbone, args, logger: logging.Logger):
    LoraConfig, _, get_peft_model = _require_peft()
    lora = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens"],
        bias="none",
        task_type=None,
    )
    model = get_peft_model(backbone, lora)
    if args.gradient_checkpointing:
        _enable_gradient_checkpointing(model, logger)
    model.config.use_cache = False
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model


def catalog_token_matrix(catalog: ItemCatalog, vocab: Vocab) -> np.ndarray:
    if catalog.num_levels != vocab.num_levels:
        raise ValueError(
            f"catalog has {catalog.num_levels} SID levels but vocab expects {vocab.num_levels}"
        )
    toks = np.empty_like(catalog.sid_codes, dtype=np.int64)
    for level, off in enumerate(vocab.sid_offsets):
        toks[:, level] = catalog.sid_codes[:, level].astype(np.int64) + off
    return toks


def build_optimizer(model: torch.nn.Module, args) -> torch.optim.Optimizer:
    embed_param_ids = set()
    layer = _embedding_layer(model)
    candidates: List[torch.nn.Module] = []
    if hasattr(layer, "modules_to_save"):
        candidates.extend(layer.modules_to_save.values())
    if hasattr(layer, "original_module"):
        candidates.append(layer.original_module)
    if not candidates and hasattr(layer, "weight"):
        embed_param_ids.add(id(layer.weight))
    for mod in list(candidates):
        embed_param_ids.update(id(p) for p in mod.parameters())

    embed_params: List[torch.nn.Parameter] = []
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in embed_param_ids:
            embed_params.append(p)
        elif p.ndim >= 2 and not any(k in name for k in ("embed", "lm_head", "norm")):
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    groups = [
        {
            "params": embed_params,
            "lr": float(args.lr) * float(args.embed_lr_mult),
            "weight_decay": 0.0,
            "name": "token_embeddings",
        },
        {"params": decay_params, "weight_decay": float(args.weight_decay), "name": "decay"},
        {"params": no_decay_params, "weight_decay": 0.0, "name": "no_decay"},
    ]
    groups = [g for g in groups if g["params"]]
    use_fused = torch.cuda.is_available()
    try:
        return torch.optim.AdamW(groups, lr=float(args.lr), betas=(0.9, 0.95),
                                 eps=1e-8, fused=use_fused)
    except TypeError:
        return torch.optim.AdamW(groups, lr=float(args.lr), betas=(0.9, 0.95), eps=1e-8)


# ---------------------------------------------------------------------------
# Run metadata (checkpoint contents)
# ---------------------------------------------------------------------------

@dataclass
class RunMeta:
    vocab: Vocab
    catalog: ItemCatalog
    user_map: UserMap
    gap_edges: np.ndarray
    model_name: str
    temperature: float
    l2_normalize: bool
    max_seq_len: int
    dtype: str
    max_feat_tokens: int = 0        # M, from the dataset manifest

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab.to_dict(), f, indent=2)
        self.catalog.save(out_dir)
        self.user_map.save(out_dir)
        np.save(os.path.join(out_dir, "gap_edges.npy"), self.gap_edges)
        with open(os.path.join(out_dir, "head_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "temperature": float(self.temperature),
                    "l2_normalize": bool(self.l2_normalize),
                    "max_seq_len": int(self.max_seq_len),
                    "max_feat_tokens": int(self.max_feat_tokens),
                    "dtype": self.dtype,
                    "num_levels": int(self.vocab.num_levels),
                    "catalog_size": int(self.catalog.size),
                    "vocab_size": int(self.vocab.total_size),
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, ckpt_dir: str) -> "RunMeta":
        with open(os.path.join(ckpt_dir, "vocab.json"), "r", encoding="utf-8") as f:
            vocab = Vocab.from_dict(json.load(f))
        with open(os.path.join(ckpt_dir, "head_config.json"), "r", encoding="utf-8") as f:
            head = json.load(f)
        catalog = ItemCatalog.load(ckpt_dir)
        user_map = UserMap.load(ckpt_dir)
        gap_edges = np.load(os.path.join(ckpt_dir, "gap_edges.npy"))
        return cls(
            vocab=vocab,
            catalog=catalog,
            user_map=user_map,
            gap_edges=gap_edges,
            model_name=head["model_name"],
            temperature=float(head["temperature"]),
            l2_normalize=bool(head["l2_normalize"]),
            max_seq_len=int(head["max_seq_len"]),
            dtype=head.get("dtype", "bf16"),
            max_feat_tokens=int(head.get("max_feat_tokens", 0) or 0),
        )




def _read_manifest(data_dir: str) -> Dict[str, Any]:
    path = os.path.join(data_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{data_dir} is not a preprocessed dataset directory ({MANIFEST_NAME} missing).\n"
            "run the preprocessing phase first, e.g.:\n"
            "  python preprocess_digr.py --sequence_path data/seq.parquet "
            "--catalog_path data/index_rqvae_prod.json --output_dir " + data_dir
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_static_meta(data_dir: str, logger: logging.Logger
                      ) -> Tuple[Vocab, ItemCatalog, UserMap, np.ndarray, Dict[str, Any]]:
    """Read the vocabulary / catalog / user map / gap edges written by preprocess_digr."""
    manifest = _read_manifest(data_dir)
    with open(os.path.join(data_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = Vocab.from_dict(json.load(f))
    catalog = ItemCatalog.load(data_dir)
    user_map = UserMap.load(data_dir)
    gap_edges = np.load(os.path.join(data_dir, "gap_edges.npy"))
    logger.info(
        f"  + loaded vocabulary from {data_dir}: total={human_int(vocab.total_size)} "
        f"(SID={vocab.sid_sizes} @ {vocab.sid_offsets}, "
        f"temporal={vocab.n_temporal} @ {vocab.offset_temporal}, "
        f"features={vocab.feature_sizes} @ {vocab.offset_features})"
    )
    logger.info(
        f"  + catalog: {human_int(catalog.size)} items x {catalog.num_levels} levels, "
        f"static user features: {len(vocab.feature_sizes)} column(s), "
        f"{human_int(manifest.get('max_feat_tokens', 0) or 0)} token(s) per row"
    )
    return vocab, catalog, user_map, gap_edges, manifest


def _resolve_meta(args, logger: logging.Logger) -> RunMeta:
    if args.resume:
        if not os.path.isdir(args.resume):
            raise FileNotFoundError(f"--resume must be a checkpoint directory: {args.resume}")
        meta = RunMeta.load(args.resume)
        if args.model_name:
            meta.model_name = args.model_name
        # Keep the exact token layout and precision the checkpoint was trained with.
        args.dtype = str(meta.dtype)
        logger.info(f"  + resuming with checkpoint metadata from {args.resume}")
        if os.path.isfile(os.path.join(args.data_dir, MANIFEST_NAME)):
            other = _read_manifest(args.data_dir)
            got = int(other.get("vocab_size", 0) or 0)
            if got and got != meta.vocab.total_size:
                raise ValueError(
                    f"{args.data_dir} was built for vocab_size={human_int(got)} but the "
                    f"checkpoint expects {human_int(meta.vocab.total_size)}; the data "
                    "and the checkpoint must come from the same catalog/gap bins"
                )
        return meta

    vocab, catalog, user_map, gap_edges, manifest = _load_static_meta(args.data_dir, logger)
    logger.info(
        f"  + temperature={args.temperature}, l2_normalize={bool(args.l2_normalize)} "
        "(both are frozen into the checkpoint's head_config)"
    )
    return RunMeta(
        vocab=vocab,
        catalog=catalog,
        user_map=user_map,
        gap_edges=gap_edges,
        model_name=args.model_name or DEFAULT_MODEL_NAME,
        temperature=float(args.temperature),
        l2_normalize=bool(args.l2_normalize),
        max_seq_len=int(manifest.get("max_seq_len") or 0),
        dtype=args.dtype,
        max_feat_tokens=int(manifest.get("max_feat_tokens") or 0),
    )


def _load_splits(args, meta: RunMeta, logger: logging.Logger
                 ) -> Tuple[TokenizedSplit, TokenizedSplit]:
    """Load the tokenised train/val CSR arrays written by preprocess_digr.py."""
    out: List[TokenizedSplit] = []
    for name in ("train", "val"):
        split = load_split(args.data_dir, name)
        if len(split) == 0:
            raise ValueError(
                f"the '{name}' split in {args.data_dir} is empty; re-run preprocessing "
                "with a lower --min_history / a wider --max_rows"
            )
        # `lengths` counts tokens: M feature tokens + SEP + (1 temporal + L SID)
        # per event.  M is a property of the user, not of the row, so derive the
        # event count after removing it.
        n_levels = max(1, meta.vocab.num_levels)
        max_feat = int(meta.max_feat_tokens)
        max_items = int((int(split.lengths.max()) - max_feat - 1) // (1 + n_levels))
        if meta.max_seq_len and max_items > meta.max_seq_len:
            logger.warning(
                f"  ! '{name}' has sequences of {max_items} events but the dataset was "
                f"prepared with --window={meta.max_seq_len} (stale digr_data?)"
            )
        logger.info(
            f"  + {name}: {human_int(len(split))} sequences, "
            f"{human_int(int(split.tokens.size))} tokens, max {max_items} items/row"
        )
        out.append(split)
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(model: DiGRRetriever, loader: DataLoader, device: torch.device,
             amp_dtype: torch.dtype, use_amp: bool) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    correct1 = 0
    correct5 = 0

    for ids, mask, targets, _ in loader:
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        b = ids.shape[0]

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            h = model.encode(ids, mask)
            logits = model.item_logits(h)
            loss = F.cross_entropy(logits, targets)

        total_loss += float(loss.item()) * b
        total_n += b
        pred = logits.argmax(dim=-1)
        correct1 += int((pred == targets).sum().item())
        k = min(5, logits.shape[1])
        topk = logits.topk(k, dim=1).indices
        correct5 += int((topk == targets.unsqueeze(1)).any(dim=1).sum().item())

    n = max(1, total_n)
    return {
        "loss": total_loss / n,
        "acc@1": correct1 / n,
        "acc@5": correct5 / n,
        "n": total_n,
    }


# ---------------------------------------------------------------------------
# Checkpoint IO
# ---------------------------------------------------------------------------

def save_checkpoint(model, meta: RunMeta, out_dir: str, state: Optional[Dict[str, Any]] = None):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    meta.save(out_dir)
    if state is not None:
        torch.save(state, os.path.join(out_dir, "train_state.pt"))


def save_vectors(model: DiGRRetriever, out_dir: str, dtype_name: str):
    # np.save has no bfloat16 dtype, so bf16 falls back to fp32 on disk.
    torch_dt = {"float16": torch.float16, "float32": torch.float32,
                "bfloat16": torch.float32}[dtype_name]
    v = model.precompute_vectors(dtype=torch_dt).cpu().numpy()
    path = os.path.join(out_dir, "catalog_vectors.npy")
    np.save(path, v)
    return path, v.shape


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def run_train(args, logger: logging.Logger):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    amp_dtype = dtype_map[args.dtype]
    use_amp = device.type == "cuda" and args.dtype != "fp32"

    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    meta = _resolve_meta(args, logger)

    log_banner(logger, "DiGR TRAINING")
    log_card(
        logger,
        "runtime configuration",
        {
            "Device": str(device),
            "Precision": f"{args.dtype} (autocast={use_amp})",
            "Backbone": meta.model_name,
            "LoRA": f"r={args.lora_r}, alpha={args.lora_alpha}, "
                    f"dropout={args.lora_dropout}, targets=q/k/v/o + embed_tokens",
            "Catalog": f"{human_int(meta.catalog.size)} items x {meta.catalog.num_levels} levels",
            "Vocab": human_int(meta.vocab.total_size),
            "Temperature": meta.temperature,
            "L2 normalise": meta.l2_normalize,
            "Max history": f"{meta.max_seq_len} events",
            "Batch / eval": f"{args.batch_size} / {args.eval_batch_size}",
            "LR (LoRA / embed)": f"{args.lr} / {args.lr * args.embed_lr_mult}",
            "Epochs / max steps": f"{args.epochs} / {args.max_steps}",
            "Resume": args.resume or "fresh",
            "Data dir": args.data_dir,
        },
    )

    train_split, val_split = _load_splits(args, meta, logger)
    train_indices = np.arange(len(train_split), dtype=np.int64)
    val_indices = np.arange(len(val_split), dtype=np.int64)
    train_lengths = train_split.lengths[train_indices]
    if len(train_lengths) < args.batch_size:
        raise ValueError(
            f"training split has {len(train_lengths)} usable sequences but "
            f"--batch_size={args.batch_size}; lower the batch size or check the data"
        )
    if len(val_indices) == 0:
        raise ValueError(
            f"the val split in {args.data_dir} is empty; re-run preprocess_digr.py"
        )

    train_ds = TokenDataset(train_split, train_indices)
    val_ds = TokenDataset(val_split, val_indices)

    logger.info(
        f"  + dataset: {human_int(len(train_ds))} train / {human_int(len(val_ds))} val, "
        f"tokens train={human_int(int(train_lengths.sum()))}"
    )

    # --- model ---------------------------------------------------------------
    backbone = load_backbone(meta.model_name, meta.vocab.total_size, args, logger)
    if args.resume:
        logger.info(f"  > loading LoRA adapter from {args.resume}")
        _, PeftModel, _ = _require_peft()
        peft_model = PeftModel.from_pretrained(backbone, args.resume, is_trainable=True)
        if args.gradient_checkpointing:
            _enable_gradient_checkpointing(peft_model, logger)
    else:
        peft_model = build_peft_backbone(backbone, args, logger)

    cat_tokens = catalog_token_matrix(meta.catalog, meta.vocab)
    retriever = DiGRRetriever(
        peft_model,
        catalog_token_ids=cat_tokens,
        temperature=meta.temperature,
        l2_normalize=meta.l2_normalize,
        logit_chunk=args.logit_chunk,
    ).to(device)

    raw_retriever = retriever
    train_forward = retriever
    if args.compile:
        logger.info("  > torch.compile(dynamic=True)")
        train_forward = torch.compile(retriever, dynamic=True)

    # --- optimisation --------------------------------------------------------
    optimizer = build_optimizer(peft_model, args)

    steps_per_epoch = max(1, len(train_ds) // args.batch_size)
    total_steps = args.max_steps if args.max_steps else args.epochs * steps_per_epoch
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return max(float(args.min_lr_ratio), 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = None
    if use_amp and amp_dtype == torch.float16:
        try:
            scaler = torch.amp.GradScaler(device.type, enabled=True)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    start_epoch = 0
    global_step = 0
    best_val = float("inf")

    state_path = os.path.join(args.resume, "train_state.pt") if args.resume else None
    if state_path and os.path.isfile(state_path):
        state = torch.load(state_path, map_location="cpu")
        start_epoch = int(state.get("epoch", 0)) + 1
        global_step = int(state.get("global_step", 0))
        best_val = float(state.get("best_val", float("inf")))
        try:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            if scaler is not None and state.get("scaler") is not None:
                scaler.load_state_dict(state["scaler"])
            logger.info(
                f"  + training state restored: epoch={start_epoch}, step={global_step}, "
                f"best_val={best_val:.4f}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  ! could not restore optimizer/scheduler state: {exc}")

    # --- loaders -------------------------------------------------------------
    collate = make_collate(meta.vocab.pad_id)
    train_sampler = LengthBucketBatchSampler(
        train_lengths, args.batch_size, shuffle=True, seed=args.seed,
        bucket_factor=args.bucket_factor, drop_last=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(4 if args.num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )

    writer = None
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=os.path.join(args.run_dir, "tb"))

    best_dir = os.path.join(args.run_dir, "best")
    final_dir = os.path.join(args.run_dir, "final")
    os.makedirs(args.run_dir, exist_ok=True)

    log_banner(logger, "TRAINING LOOP")
    logger.info(
        f"  steps/epoch={steps_per_epoch} total_steps={total_steps} "
        f"warmup={warmup_steps} grad_accum={args.grad_accum}"
    )

    stop = False
    epoch = start_epoch - 1          # guards the zero-epoch / resume-past-end case
    final_saved_epoch = -1
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        train_sampler.set_epoch(epoch)
        model = train_forward
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_t0 = time.time()
        micro = 0
        running_loss = 0.0
        running_acc = 0.0
        window = 0
        tokens_seen = 0
        pending = False
        pending_micros = 0

        for ids, mask, targets, _ in train_loader:
            ids = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            tokens_seen += int(mask.sum().item())

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                h = model(ids, mask)
                logits = model.item_logits(h)
                loss = F.cross_entropy(
                    logits, targets, label_smoothing=float(args.label_smoothing)
                )

            scaled_loss = loss / float(args.grad_accum)
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            micro += 1
            pending = True
            pending_micros += 1
            with torch.no_grad():
                running_loss += float(loss.item())
                running_acc += float((logits.argmax(dim=-1) == targets).float().mean().item())
                window += 1

            if micro % args.grad_accum != 0:
                continue

            if scaler is not None:
                scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in retriever.parameters() if p.requires_grad], args.grad_clip
                )
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pending = False
            pending_micros = 0
            global_step += 1

            if global_step % args.log_interval == 0:
                elapsed = max(1e-6, time.time() - epoch_t0)
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"  ep {epoch + 1}/{args.epochs} | step {global_step} | "
                    f"loss {running_loss / max(1, window):.4f} | "
                    f"acc@1 {running_acc / max(1, window):.4f} | "
                    f"lr {lr:.2e} | {tokens_seen / elapsed:,.0f} tok/s"
                )
                if writer is not None:
                    writer.add_scalar("train/loss", running_loss / max(1, window), global_step)
                    writer.add_scalar("train/acc1", running_acc / max(1, window), global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    writer.add_scalar("train/tokens_per_s", tokens_seen / elapsed, global_step)
                running_loss = 0.0
                running_acc = 0.0
                window = 0

            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break
        # flush a partial accumulation window
        if pending:
            # Scale up so the tail step is worth one full --grad_accum window.
            if pending_micros < args.grad_accum:
                tail_scale = float(args.grad_accum) / float(max(1, pending_micros))
                for p in retriever.parameters():
                    if p.grad is not None:
                        p.grad.mul_(tail_scale)
            if scaler is not None:
                scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in retriever.parameters() if p.requires_grad], args.grad_clip
                )
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        val = evaluate(raw_retriever, val_loader, device, amp_dtype, use_amp)
        if window:
            logger.info(
                f"  ep {epoch + 1} train tail | loss {running_loss / window:.4f} | "
                f"acc@1 {running_acc / window:.4f}"
            )
        logger.info(
            f"  == epoch {epoch + 1} done in {(time.time() - epoch_t0) / 60:.1f} min | "
            f"val loss {val['loss']:.4f} | acc@1 {val['acc@1']:.4f} | "
            f"acc@5 {val['acc@5']:.4f}"
        )
        if writer is not None:
            writer.add_scalar("val/loss", val["loss"], epoch + 1)
            writer.add_scalar("val/acc1", val["acc@1"], epoch + 1)
            writer.add_scalar("val/acc5", val["acc@5"], epoch + 1)

        improved = val["loss"] < best_val
        if improved:
            best_val = val["loss"]
            save_checkpoint(
                peft_model, meta, best_dir,
                state={
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val": best_val,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                },
            )
            logger.info(f"  + new best checkpoint -> {best_dir} (val loss {best_val:.4f})")

        if (epoch + 1) % max(1, args.save_interval) == 0 or stop:
            save_checkpoint(
                peft_model, meta, final_dir,
                state={
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val": best_val,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                },
            )
            logger.info(f"  + checkpoint -> {final_dir}")
            final_saved_epoch = epoch

    # --- final artefacts (Phase 4 inference cache) ---------------------------
    if final_saved_epoch != epoch or not os.path.isdir(final_dir):
        save_checkpoint(
            peft_model, meta, final_dir,
            state={
                "epoch": epoch,
                "global_step": global_step,
                "best_val": best_val,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
            },
        )
    path, shape = save_vectors(raw_retriever, final_dir, args.vector_dtype)
    logger.info(f"  + cached catalog vectors {shape} -> {path}")
    if writer is not None:
        writer.flush()
        writer.close()
    log_banner(logger, f"DONE - best val loss {best_val:.4f}")


# ---------------------------------------------------------------------------
# Inference / export
# ---------------------------------------------------------------------------

def load_for_inference(args, logger: logging.Logger):
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for infer/export")

    meta = RunMeta.load(args.checkpoint)
    args.dtype = str(meta.dtype)          # match the precision used for training
    model_name = args.model_name or meta.model_name

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    amp_dtype = dtype_map[args.dtype]
    use_amp = device.type == "cuda" and args.dtype != "fp32"

    backbone = load_backbone(model_name, meta.vocab.total_size, args, logger)
    _, PeftModel, _ = _require_peft()
    peft_model = PeftModel.from_pretrained(backbone, args.checkpoint, is_trainable=False)
    peft_model.eval()

    cat_tokens = catalog_token_matrix(meta.catalog, meta.vocab)
    retriever = DiGRRetriever(
        peft_model,
        catalog_token_ids=cat_tokens,
        temperature=meta.temperature,
        l2_normalize=meta.l2_normalize,
        logit_chunk=args.logit_chunk,
    ).to(device)
    retriever.eval()
    return meta, retriever, device, amp_dtype, use_amp


def write_predictions(path: str, users: np.ndarray, topk_idx: np.ndarray,
                      topk_scores: np.ndarray, catalog: ItemCatalog):
    item_ids = catalog.item_ids
    out_ids = [[item_ids[int(i)] for i in row] for row in topk_idx]
    user_vals = users.tolist() if isinstance(users, np.ndarray) else list(users)
    table = pa.table(
        {
            "user_id": pa.array(user_vals),
            "topk_item_ids": pa.array(out_ids),
            "topk_scores": pa.array(
                topk_scores.astype(np.float32).tolist(), type=pa.list_(pa.float32())
            ),
        }
    )
    pq.write_table(table, path, compression="zstd")
    return len(user_vals)


@torch.inference_mode()
def run_infer(args, logger: logging.Logger):
    meta, retriever, device, amp_dtype, use_amp = load_for_inference(args, logger)
    log_banner(logger, "DiGR INFERENCE")
    logger.info(
        f"  catalog={human_int(meta.catalog.size)} | vocab={human_int(meta.vocab.total_size)} | "
        f"tau={retriever.temperature} | topk={args.topk}"
    )

    # Phase 4 step 1: V is computed once and frozen.
    vectors_path = os.path.join(args.checkpoint, "catalog_vectors.npy")
    if args.use_saved_vectors and os.path.isfile(vectors_path):
        vectors = torch.from_numpy(np.load(vectors_path)).to(device)
        if vectors.dtype != torch.bfloat16 and device.type == "cuda":
            vectors = vectors.to(torch.bfloat16)
        logger.info(f"  + loaded cached catalog vectors {tuple(vectors.shape)}")
    else:
        vectors = retriever.precompute_vectors(dtype=torch.bfloat16)
        logger.info(f"  + precomputed catalog vectors {tuple(vectors.shape)}")

    # NOTE: the preprocessed history ends *before* the held-out target for every
    # row of the `val`/`test` splits, so retrieving on those splits reproduces the
    # offline metrics.  For online serving, re-tokenise the live history yourself
    # (the newest interaction must be part of it) and pass it as a `--split` file.
    split_name = args.infer_split
    split = load_split(args.data_dir, split_name)
    if len(split) == 0:
        raise ValueError(
            f"split '{split_name}' in {args.data_dir} is empty; available: "
            + ", ".join(s for s in ("train", "val", "infer") if split_exists(args.data_dir, s))
        )
    indices = np.arange(len(split), dtype=np.int64)
    if args.max_rows is not None:
        indices = indices[: max(1, int(args.max_rows))]
    logger.info(
        f"  + {split_name} split: retrieving for {human_int(len(indices))} of "
        f"{human_int(len(split))} sequences"
    )
    ds = TokenDataset(split, indices)
    loader = DataLoader(
        ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=make_collate(meta.vocab.pad_id),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_idx: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_users: List[np.ndarray] = []
    t0 = time.time()
    n_rows = 0

    for ids, mask, targets, rows in loader:
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            h = retriever.encode(ids, mask)
        top_idx, top_scores = retriever.retrieve(
            h, vectors, args.topk, args.infer_catalog_chunk
        )
        all_idx.append(top_idx.cpu().numpy())
        all_scores.append(top_scores.float().cpu().numpy())
        all_users.append(split.users[rows.numpy()])
        n_rows += ids.shape[0]

    if not all_idx:
        raise ValueError("inference produced no rows")

    topk_idx = np.concatenate(all_idx, axis=0)
    topk_scores = np.concatenate(all_scores, axis=0)
    users = np.concatenate(all_users, axis=0)

    out_path = args.infer_output or os.path.join(args.run_dir, "predictions.parquet")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n_written = write_predictions(out_path, users, topk_idx, topk_scores, meta.catalog)
    elapsed = max(1e-6, time.time() - t0)
    logger.info(
        f"  + wrote {human_int(n_written)} rows x top-{args.topk} -> {out_path} "
        f"({n_written / elapsed:,.0f} rows/s)"
    )


def run_export(args, logger: logging.Logger):
    meta, retriever, device, _, _ = load_for_inference(args, logger)
    out_dir = args.run_dir or args.checkpoint
    os.makedirs(out_dir, exist_ok=True)
    path, shape = save_vectors(retriever, out_dir, args.vector_dtype)
    logger.info(f"  + exported catalog vectors {shape} -> {path}")
    with open(os.path.join(out_dir, "catalog_vectors_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_items": int(shape[0]),
                "dim": int(shape[1]),
                "dtype": args.vector_dtype,
                "temperature": float(retriever.temperature),
                "l2_normalize": bool(retriever.l2_normalize),
                "catalog_size": int(meta.catalog.size),
            },
            f,
            indent=2,
        )
    logger.info(
        "  hint: vectors are mean-pooled SID embeddings; for FAISS convert to "
        "float32 and use IndexFlatIP (L2-normalise first if --l2_normalize)."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DiGR: Qwen + LoRA generative retrieval over RQ-VAE Semantic IDs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--mode", choices=["train", "infer", "export"], default="train")
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                   help="preprocessed dataset directory written by preprocess_digr.py")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="checkpoint dir to resume training")
    p.add_argument("--checkpoint", type=str, default=None, help="checkpoint dir for infer/export")
    p.add_argument("--infer_output", type=str, default=None)

    # dataset-independent knobs
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--bucket_factor", type=int, default=50)
    p.add_argument("--seed", type=int, default=SEED)

    # model
    p.add_argument("--model_name", type=str, default=None,
                   help=f"HF model id (default: {DEFAULT_MODEL_NAME}, or the checkpoint's)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--attn_implementation", type=str, default=None,
                   help="flash_attention_2 | sdpa | eager (default: auto-fallback)")
    p.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=False,
                   help="allow custom HF modelling code (Qwen does not need it)")
    p.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R)
    p.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA)
    p.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reinit_embeddings", action=argparse.BooleanOptionalAction, default=True,
                   help="draw N(0,0.02) for the resized token table (every id in the "
                        "custom vocab is a new SID/temporal/feature token, so the "
                        "truncated Qwen text rows carry no useful semantics)")
    p.add_argument("--l2_normalize", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--logit_chunk", type=int, default=DEFAULT_LOGIT_CHUNK)
    p.add_argument("--compile", action="store_true")

    # optimisation
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--eval_batch_size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--embed_lr_mult", type=float, default=DEFAULT_EMBED_LR_MULT)
    p.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    p.add_argument("--min_lr_ratio", type=float, default=DEFAULT_MIN_LR_RATIO)
    p.add_argument("--grad_clip", type=float, default=DEFAULT_GRAD_CLIP)
    p.add_argument("--label_smoothing", type=float, default=DEFAULT_LABEL_SMOOTHING)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--save_interval", type=int, default=1)

    # inference
    p.add_argument("--infer_split", choices=["train", "val", "infer"], default="val",
                   help="which preprocessed split to run retrieval over (val = held-out "
                        "target removed from history; infer = whole-history rows written "
                        "by preprocess_digr.py --infer_sequence_path)")
    p.add_argument("--max_rows", type=int, default=None,
                   help="inference only: cap on the number of sequences")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--infer_catalog_chunk", type=int, default=200_000)
    p.add_argument("--use_saved_vectors", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vector_dtype", choices=["float16", "float32"], default="float16")

    args = p.parse_args(argv)

    if args.mode != "train" and not args.checkpoint:
        p.error("--checkpoint is required for infer/export")
    if args.grad_accum < 1:
        p.error("--grad_accum must be >= 1")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        p.error("--batch_size and --eval_batch_size must be >= 1")
    if args.log_interval < 1 or args.save_interval < 1:
        p.error("--log_interval and --save_interval must be >= 1")
    if args.topk < 1 or args.infer_catalog_chunk < 1 or args.logit_chunk < 1:
        p.error("--topk, --infer_catalog_chunk and --logit_chunk must be >= 1")
    if args.bucket_factor < 1 or args.num_workers < 0:
        p.error("--bucket_factor must be >= 1 and --num_workers >= 0")
    if args.max_rows is not None and args.max_rows < 1:
        p.error("--max_rows must be >= 1")
    return args


def _validate_cli_paths(args) -> None:
    """Fail fast with a readable message instead of a pyarrow traceback."""
    if args.mode in ("infer", "export"):
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --mode infer/export")
        if not os.path.isdir(args.checkpoint):
            raise SystemExit(f"--checkpoint must be a run directory: {args.checkpoint}")
    if args.mode in ("train", "infer"):
        if not os.path.isdir(args.data_dir):
            raise SystemExit(
                f"preprocessed dataset not found: {args.data_dir}\n"
                "run the preprocessing phase first, e.g.:\n"
                f"  python preprocess_digr.py --sequence_path data/seq.parquet "
                f"--catalog_path data/index_rqvae_prod.json --output_dir {args.data_dir}"
            )
        missing = [n for n in (MANIFEST_NAME, "vocab.json", "catalog_item_ids.npy",
                              "users.npy", "gap_edges.npy")
                   if not os.path.exists(os.path.join(args.data_dir, n))]
        if missing:
            raise SystemExit(
                f"{args.data_dir} is incomplete (missing: {', '.join(missing)}); "
                "re-run preprocess_digr.py"
            )
        need = ["train", "val"] if args.mode == "train" else [args.infer_split]
        for name in need:
            if not split_exists(args.data_dir, name):
                raise SystemExit(
                    f"split directory '{name}' missing in {args.data_dir}; "
                    "re-run preprocess_digr.py"
                )


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    _validate_cli_paths(args)

    if args.mode == "train":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.run_name:
            name = args.run_name
        elif args.resume:
            name = f"digr_resume_{timestamp}"
        else:
            name = f"digr_{timestamp}"
        args.run_dir = os.path.join(args.output_dir, name)
        log_dir = os.path.join(args.run_dir, "logs")
    else:
        args.run_dir = args.checkpoint
        log_dir = args.checkpoint

    logger = setup_logger(log_dir)
    try:
        if args.mode == "train":
            run_train(args, logger)
        elif args.mode == "infer":
            run_infer(args, logger)
        else:
            run_export(args, logger)
    except Exception:
        logger.exception("fatal error")
        raise


if __name__ == "__main__":
    main()
