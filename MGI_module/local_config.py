"""Repository-relative defaults for the local Qwen deployment."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("MODEL_PATH", PROJECT_ROOT / "models/Qwen2.5-7B-Instruct"))
EMBEDDING_MODEL_PATH = Path(os.environ.get(
    "EMBEDDING_MODEL_PATH", PROJECT_ROOT / "sentence-transformers/all-MiniLM-L6-v2"
))
PERSONAMEM_DIR = Path(os.environ.get("DATASET_ROOT", PROJECT_ROOT / "MGI_module/datasets/PersonaMem"))
MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "Qwen2.5-7B-Instruct")
