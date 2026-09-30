"""Check model routing and optionally the local judge's JSON response."""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("VLLM_URL", "http://127.0.0.1:6025/v1"))
    parser.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "Qwen2.5-7B-Instruct"))
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()
    key = os.environ.get("MEMCOE_API_KEY", "memcoe-local")
    request = urllib.request.Request(args.url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=15) as response:
        model_ids = {item["id"] for item in json.load(response)["data"]}
    if args.model not in model_ids:
        raise RuntimeError(f"Expected model {args.model}; available models: {sorted(model_ids)}")
    print(f"Model service ready: {args.model} at {args.url}")
    if args.judge:
        from recurrent.intermediate_reward import OpenAIMemoryRewardJudge
        os.environ.setdefault("MEMCOE_API_KEY", key)
        result = OpenAIMemoryRewardJudge(model=args.model, base_url=args.url, temperature=0).score(
            "What drink does the user prefer?", "No previous memory",
            "User: I prefer tea to coffee.", "Drink preference: tea over coffee.")
        print(f"Judge JSON check passed: score={result.score}, reason={result.reason}")


if __name__ == "__main__":
    main()
