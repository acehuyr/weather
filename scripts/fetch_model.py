"""Download only the PyTorch files of the sentence-transformers model.

A plain `SentenceTransformer("all-MiniLM-L6-v2")` pulls the full repository -
ONNX, OpenVINO and TensorFlow variants included - which is roughly a gigabyte
for a model whose PyTorch weights are about 90 MB. This fetches just what the
PyTorch path needs.

    python scripts/fetch_model.py
"""
from __future__ import annotations

import sys

from huggingface_hub import snapshot_download

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Everything SentenceTransformer touches on the PyTorch path.
ALLOW = [
    "*.json",
    "*.txt",
    "model.safetensors",
    "1_Pooling/*",
]

# The large duplicates in other runtimes' formats.
IGNORE = [
    "onnx/*",
    "openvino/*",
    "rust_model.ot",
    "tf_model.h5",
    "model.onnx",
    "pytorch_model.bin",   # superseded by model.safetensors
]


def main() -> int:
    print(f"Fetching {MODEL} (PyTorch files only)...")
    path = snapshot_download(
        MODEL, allow_patterns=ALLOW, ignore_patterns=IGNORE, max_workers=4
    )
    print(f"Done: {path}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL)
    vectors = model.encode(["monsoon onset over Kerala"], normalize_embeddings=True)
    print(f"Loaded OK - dim {model.get_sentence_embedding_dimension()}, "
          f"norm {float((vectors[0] ** 2).sum()) ** 0.5:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
