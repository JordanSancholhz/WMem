"""Prepare disjoint question splits and Top-K context Parquet files for MemCoE.

Without an author-supplied ID file, seed 42 selects 300 of the 589 32K questions.
This is a new question-level split, not a reconstruction of the paper's split.
"""
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from MGI_module.local_config import PERSONAMEM_DIR, EMBEDDING_MODEL_PATH, MODEL_PATH


def select_split(question_ids, train_count=300, seed=42, supplied_ids=None):
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Duplicate question IDs in input data")
    if supplied_ids is None:
        if not 0 < train_count < len(question_ids):
            raise ValueError("train_count must leave non-empty train and test splits")
        train_ids = set(random.Random(seed).sample(question_ids, train_count))
    else:
        if not isinstance(supplied_ids, list) or len(supplied_ids) != len(set(supplied_ids)):
            raise ValueError("Training ID file must be a JSON list without duplicates")
        train_ids = set(supplied_ids)
        if train_ids - set(question_ids):
            raise ValueError("Training ID file contains unknown question IDs")
        if not train_ids or len(train_ids) == len(question_ids):
            raise ValueError("Train and test splits must both be non-empty")
    return ([q for q in question_ids if q in train_ids], [q for q in question_ids if q not in train_ids])


def build_record(item, contexts, index, split):
    context = "\n\n".join(contexts)
    if not context.strip():
        raise ValueError(f"Empty retrieved context for {item.question_id}")
    return {
        "data_source": "personamem",
        "prompt": [{"role": "user", "content": f"<question> {item.question} </question>\n\n<options> {item.options} </options>"}],
        "context": context,
        "ability": "personalization",
        "reward_model": {"style": "rule", "ground_truth": item.correct_answer},
        "extra_info": {"index": index, "question_id": item.question_id, "split": split,
                       "question_type": item.question_type or "", "persona_id": item.persona_id or ""},
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=PERSONAMEM_DIR)
    parser.add_argument("--output_dir", type=Path, default=PERSONAMEM_DIR)
    parser.add_argument("--size", choices=["32k", "128k", "1M"], default="32k")
    parser.add_argument("--train_ids", type=Path, help="Author-supplied training question IDs, if available")
    parser.add_argument("--train_count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--embedding_model", type=Path, default=EMBEDDING_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0", help="Embedding device; CUDA_VISIBLE_DEVICES remaps physical IDs")
    parser.add_argument("--split_only", action="store_true", help="Validate raw data and write IDs without loading models")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--recommended-max-chunks", type=Path, help="Read prepared context lengths; print the required number of 512-token chunks")
    args = parser.parse_args()
    if args.recommended_max_chunks:
        manifest = json.loads(args.recommended_max_chunks.read_text(encoding="utf-8"))
        if not manifest.get("prepared"):
            raise ValueError("Run full preprocessing before training; split-only output has no context lengths")
        print(max(8, math.ceil(manifest["max_context_tokens"] / 512)))
        return

    from MGI_module.loaders.personamem import load_personamem
    items = load_personamem(str(args.data_dir), args.size)
    output = args.output_dir.resolve()
    train_file = output / f"RAG_top{args.topk}_{args.size}_train.parquet"
    test_file = output / f"RAG_top{args.topk}_{args.size}_test.parquet"
    if (train_file.exists() or test_file.exists()) and not args.overwrite:
        raise FileExistsError("Prepared files already exist; use --overwrite to replace them")
    ids_path = args.train_ids or output / "question_id_train_all.json"
    supplied_ids = json.loads(ids_path.read_text(encoding="utf-8")) if ids_path.exists() else None
    if args.train_ids and supplied_ids is None:
        raise FileNotFoundError(args.train_ids)
    train_ids, test_ids = select_split([item.question_id for item in items], args.train_count, args.seed, supplied_ids)
    train_set = set(train_ids)
    manifest = {
        "size": args.size, "seed": args.seed, "topk": args.topk,
        "split_source": str(ids_path.resolve()) if supplied_ids is not None else "generated_question_split",
        "split_note": "Question-level split; histories/personas may overlap. Not a verified reproduction of the paper's split.",
        "train_questions": len(train_ids), "test_questions": len(test_ids),
        "source_files": {name: sha256(args.data_dir / name) for name in [f"questions_{args.size}.csv", f"shared_contexts_{args.size}.jsonl"]},
        "embedding_model": str(args.embedding_model.resolve()), "tokenizer": str(MODEL_PATH.resolve()),
        "prepared": False,
    }
    if not args.split_only:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from MGI_module.processors.rag import RAGProcessor
        from MGI_module.tokenization import get_tokenizer
        from tqdm import tqdm
        retriever = RAGProcessor(str(args.embedding_model), topk=args.topk, device=args.device)
        tokenizer = get_tokenizer()
        records = {"train": [], "test": []}
        max_tokens = 0
        for index, item in enumerate(tqdm(items, desc="Retrieving PersonaMem contexts")):
            split = "train" if item.question_id in train_set else "test"
            record = build_record(item, retriever.retrieve(item.question, item.context), index, split)
            max_tokens = max(max_tokens, len(tokenizer.encode(record["context"], add_special_tokens=False)))
            records[split].append(record)
        output.mkdir(parents=True, exist_ok=True)
        for split, destination in [("train", train_file), ("test", test_file)]:
            table = pa.Table.from_pylist(records[split])
            temporary = destination.with_suffix(".parquet.tmp")
            pq.write_table(table, temporary)
            if pq.read_metadata(temporary).num_rows != len(records[split]):
                raise RuntimeError(f"Parquet row count mismatch: {temporary}")
            temporary.replace(destination)
        manifest.update(prepared=True, max_context_tokens=max_tokens,
                        train_file=str(train_file), test_file=str(test_file))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "question_id_train_all.json", train_ids)
    write_json(output / "question_id_test_all.json", test_ids)
    write_json(output / "preparation_manifest.json", manifest)
    print(f"Train: {len(train_ids)}, test: {len(test_ids)}, output: {output}")
    print(manifest["split_note"])


if __name__ == "__main__":
    main()
