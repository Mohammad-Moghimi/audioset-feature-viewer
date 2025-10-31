
---

### `view_audioset_feature.py`


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Read and display one SequenceExample from an AudioSet embeddings .tfrecord shard.

Context features:
- video_id: bytes (YouTube ID)
- start_time_seconds: float
- end_time_seconds: float
- labels: int64 list

Feature lists:
- audio_embedding: sequence of bytes; each is 128 uint8 values (one per ~0.96 s frame)
"""

import argparse
import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


def list_shards(data_dir: str) -> List[str]:
    shards = sorted(glob.glob(os.path.join(data_dir, "*.tfrecord")))
    if not shards:
        raise FileNotFoundError(f"No .tfrecord files found in: {data_dir}")
    return shards


def load_label_map(csv_path: str) -> Dict[int, Tuple[str, str]]:
    """
    Returns: dict[index] -> (mid, display_name)
    CSV columns: index, mid, display_name
    """
    label_map: Dict[int, Tuple[str, str]] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index"])
            label_map[idx] = (row["mid"], row["display_name"])
    return label_map


def parse_one_example(record_bytes: bytes):
    seq = tf.train.SequenceExample()
    seq.ParseFromString(record_bytes)

    ctx = seq.context.feature
    vid = ctx["video_id"].bytes_list.value[0].decode("utf-8")
    start = float(ctx["start_time_seconds"].float_list.value[0])
    end = float(ctx["end_time_seconds"].float_list.value[0])
    labels = list(ctx["labels"].int64_list.value)

    feats = seq.feature_lists.feature_list["audio_embedding"].feature
    # Convert each frame from 128-byte string -> (128,) uint8 numpy array
    embeddings = np.vstack(
        [np.frombuffer(f.bytes_list.value[0], dtype=np.uint8) for f in feats]
    )  # shape (num_frames, 128)

    return vid, start, end, labels, embeddings


def read_example_from_shard(
    shard_path: str, example_index: int
):
    ds = tf.data.TFRecordDataset(shard_path)
    for i, raw in enumerate(ds):
        if i == example_index:
            return parse_one_example(raw.numpy())
    raise IndexError(
        f"example_index {example_index} out of range for shard {os.path.basename(shard_path)}"
    )


def format_labels(
    label_indices: List[int], label_map: Optional[Dict[int, Tuple[str, str]]]
) -> str:
    if not label_indices:
        return "[]"
    if label_map is None:
        return str(label_indices)
    names = []
    for idx in label_indices:
        if idx in label_map:
            mid, name = label_map[idx]
            names.append(f"{name} ({idx},{mid})")
        else:
            names.append(f"<unknown:{idx}>")
    return "[" + ", ".join(names) + "]"


def main():
    parser = argparse.ArgumentParser(
        description="Inspect one example from AudioSet VGGish embedding shards."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing .tfrecord shards (e.g., .../audioset_v1_embeddings/eval)",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Which shard in the sorted list to read (default: 0)",
    )
    parser.add_argument(
        "--example_index",
        type=int,
        default=0,
        help="Which example within the shard to read (default: 0)",
    )
    parser.add_argument(
        "--frames_to_show",
        type=int,
        default=3,
        help="How many initial frames to print (default: 3)",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default=None,
        help="Path to AudioSet class_labels_indices.csv to map indices to names (optional)",
    )
    args = parser.parse_args()

    shards = list_shards(args.data_dir)
    shard_path = shards[args.shard_index]

    label_map = load_label_map(args.labels_csv) if args.labels_csv else None

    vid, start, end, labels, embeddings = read_example_from_shard(
        shard_path, args.example_index
    )

    print(f"File: {shard_path}")
    print(f"Example index: {args.example_index}")
    print(f"video_id: {vid}")
    print(f"segment: {start:.2f}s → {end:.2f}s  (duration {end-start:.2f}s)")
    print(f"labels (indices): {labels}")
    if label_map is not None:
        print(f"labels (mapped): {format_labels(labels, label_map)}")

    print(f"embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    n = max(0, min(args.frames_to_show, embeddings.shape[0]))
    for i in range(n):
        print(f"frame {i}: {embeddings[i]}")
    if embeddings.shape[0] > n:
        print(f"... ({embeddings.shape[0]-n} more frames omitted)")


if __name__ == "__main__":
    # Disable TF logs for a cleaner CLI
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
