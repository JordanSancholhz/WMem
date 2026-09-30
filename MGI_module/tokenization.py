"""Use the supplied Qwen tokenizer; no separate tiktoken download is needed."""
from functools import lru_cache
from MGI_module.local_config import MODEL_PATH


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Local Qwen model not found: {MODEL_PATH}")
    return AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)


def split_context(context, tokenizer, num_chunks=8, chunk_size=None):
    if num_chunks < 1 or (chunk_size is not None and chunk_size < 1):
        raise ValueError("num_chunks and chunk_size must be positive")
    tokens = tokenizer.encode("\n\n".join(context), add_special_tokens=False)
    width = chunk_size or max(1, (len(tokens) + num_chunks - 1) // num_chunks)
    return [tokenizer.decode(tokens[i:i + width]) for i in range(0, len(tokens), width)] or [""]
