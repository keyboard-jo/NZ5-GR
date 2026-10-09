import json
from collections import Counter
import h5py
import numpy as np

with open('local/2026-09-10-1/test_output/index_rqvae_prod.json') as f:
    mapping = json.load(f)

# Find the most frequent 4-code tuple
tuple_counts = Counter(tuple(v[:4]) for v in mapping.values())
worst_tuple, count = tuple_counts.most_common(1)[0]
print(f"Worst tuple {worst_tuple} appears {count} times.")

# Get IDs in this tuple
items_in_worst = [k for k, v in mapping.items() if tuple(v[:4]) == worst_tuple]

# Check their raw embeddings in the H5 file
with h5py.File('data/mm_emb_82.h5', 'r') as f:
    sample_embs = f['embedding'][:len(items_in_worst)]
    std_dev = np.std(sample_embs, axis=0).mean()
    print(f"Mean feature std dev in worst cluster: {std_dev:.6f}")