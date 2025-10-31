# AudioSet Feature Viewer

Small utility to inspect a single example from the **AudioSet VGGish embeddings** (`.tfrecord` shards) published at:
https://research.google.com/audioset/download.html

Each record is a TensorFlow `SequenceExample` with:
- **context**: `video_id` (bytes), `start_time_seconds` (float), `end_time_seconds` (float), `labels` (list[int])
- **feature_lists**: `audio_embedding` — per-frame 128-D embeddings as 128 uint8 bytes (≈0.96 s hop)

This script opens one shard, reads one example, prints the metadata, and shows the first few embeddings.

---

## Contents

- `view_audioset_feature.py` — CLI to read and display one example.
- `requirements.txt` — Python deps.
- `.gitignore` — Housekeeping.

---

## Installation

```bash
# Python 3.9+ recommended
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
