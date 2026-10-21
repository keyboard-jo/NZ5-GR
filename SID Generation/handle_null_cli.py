import argparse
import numpy as np
import polars as pl

# 1. Set up command-line argument parser
parser = argparse.ArgumentParser(
    description="Clean and impute missing values in categorical and multimodal item features."
)
parser.add_argument(
    "--input",
    type=str,
    default="data/canonical_items.parquet",
    help="Path to the input parquet file.",
)
parser.add_argument(
    "--output",
    type=str,
    default="canonical_items_cleaned.parquet",
    help="Path to save the cleaned parquet file.",
)
args = parser.parse_args()

print(f"Loading dataset from: {args.input}")
df = pl.read_parquet(args.input)

# Define categorical columns and multimodal column
cat_cols = [
    "100",
    "101",
    "102",
    "112",
    "114",
    "115",
    "116",
    "117",
    "118",
    "119",
    "120",
    "121",
    "122",
]
mm_col = "mm_emb"
flag_col = "has_mm_emb"

# 2. Define a 1024-dimensional zero vector for missing multimodal embeddings
ZERO_VEC_1024 = [0.0] * 1024

# 3. Clean and impute missing values
cleaned_df = df.with_columns([
    # --- A. Synchronize the missingness flag ---
    (pl.col(mm_col).is_not_null() & pl.col(flag_col).fill_null(False)).alias(
        flag_col
    ),
    # --- B. Zero-fill missing 1024-D embeddings ---
    pl.when(pl.col(mm_col).is_not_null())
    .then(pl.col(mm_col))
    .otherwise(pl.lit(ZERO_VEC_1024, dtype=pl.Array(pl.Float32, 1024)))
    .alias(mm_col),
    # --- C. Handle nulls in categorical columns ---
    # Fill nulls with 0 (reserved index for <MISSING> / <UNK>)
    *[pl.col(c).fill_null(0).alias(c) for c in cat_cols],
])

# 4. Verify there are NO nulls left anywhere in the dataset
print("\n=== Null Counts After Imputation ===")
print(cleaned_df.null_count())

# 5. Save the cleaned dataset
cleaned_df.write_parquet(args.output)
print(f"\nSuccessfully saved to {args.output}!")