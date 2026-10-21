import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import polars as pl

def parse_args():
    parser = argparse.ArgumentParser(description="Map HDF5 embeddings to canonical items using OID-to-RID mapping.")
    parser.add_argument(
        "--item-feat-path",
        type=str,
        default=str(Path.home() / "NZ5" / "make_data" / "item_feat.parquet"),
        help="Path to the item features parquet file."
    )
    parser.add_argument(
        "--i-json-path",
        type=str,
        default=str(Path.home() / "NZ5" / "make_data" / "i.json"),
        help="Path to the i.json OID to RID mapping file."
    )
    parser.add_argument(
        "--h5-emb-path",
        type=str,
        default="/scratch/mscbdtspod2026/jksulaiman/data/mm_emb_82.h5",
        help="Path to the HDF5 embeddings file."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(Path.home() / "NZ5" / "make_data" / "canonical_items.parquet"),
        help="Path to save the output master dataset parquet file."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    ITEM_FEAT_PATH = Path(args.item_feat_path)
    I_JSON_PATH = Path(args.i_json_path)
    H5_EMB_PATH = Path(args.h5_emb_path)
    OUTPUT_PATH = Path(args.output_path)

    # ==========================================
    # 1. Load Data Sources
    # ==========================================
    print(f"Loading item features from: {ITEM_FEAT_PATH}")
    item_feat = pl.read_parquet(ITEM_FEAT_PATH)

    print(f"Loading OID -> RID mapping from: {I_JSON_PATH}")
    with open(I_JSON_PATH, "r") as f:
        i_mapping = json.load(f)

    oid_to_rid = pl.DataFrame({
        "anonymous_cid": list(i_mapping.keys()),
        "item_id": list(i_mapping.values()),
    }).with_columns(
        pl.col("anonymous_cid").cast(pl.String),
        pl.col("item_id").cast(pl.Int64),
    )

    print(f"Loading embeddings from HDF5: {H5_EMB_PATH}")
    with h5py.File(H5_EMB_PATH, "r") as f:
        cids = f["anonymous_cid"][:]
        if cids.dtype.kind == "b":
            cids = np.array([c.decode("utf-8") for c in cids])
            
        embeddings = f["embedding"][:]

    mm_82 = pl.DataFrame({
        "anonymous_cid": cids,
        "mm_emb": list(embeddings),
    }).with_columns(
        pl.col("anonymous_cid").cast(pl.String)
    )

    # ==========================================
    # 2. Convert OID to RID for Embeddings
    # ==========================================
    print("Mapping OID embeddings to RID...")
    mm_by_rid = (
        mm_82
        .join(
            oid_to_rid,
            on="anonymous_cid",
            how="inner",
        )
        .select(
            "item_id",
            "mm_emb",
        )
    )

    # ==========================================
    # 3. Left Outer Join onto Canonical Item Universe
    # ==========================================
    print("Performing left outer join onto canonical item universe...")
    canonical_items = (
        item_feat
        .join(
            mm_by_rid,
            on="item_id",
            how="left",
        )
        .with_columns(
            pl.col("mm_emb")
            .is_not_null()
            .alias("has_mm_emb")
        )
    )

    # ==========================================
    # 4. Persist Master Dataset
    # ==========================================
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving master dataset to: {OUTPUT_PATH}")
    canonical_items.write_parquet(OUTPUT_PATH)
    print("Done!")

if __name__ == "__main__":
    main()