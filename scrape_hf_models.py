"""
scripts/scrape_hf_models.py
============================
Fetches model metadata from the HuggingFace API and rebuilds models_db.json.

This is a more powerful version of the built-in --update flag. Use this script
when you want more control over what gets fetched — for example, searching for
a specific model family or setting a custom rate-limit delay.

Uses only Python standard library — no pip installs required.

Usage:
    python scripts/scrape_hf_models.py                  # Fetch all models
    python scripts/scrape_hf_models.py --search qwen    # Only Qwen models
    python scripts/scrape_hf_models.py --limit 20       # First 20 only
    python scripts/scrape_hf_models.py --delay 0.5      # Slower (safer) rate
"""

import json
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime


# Output path — always saves next to the parent checker.py
OUTPUT_PATH = Path(__file__).parent.parent / "models_db.json"
HF_API_BASE = "https://huggingface.co/api"


# ---------------------------------------------------------------------------
# Quantization memory fractions (same values as checker.py)
# ---------------------------------------------------------------------------

QUANT_MEMORY_FRACTION = {
    "Q8_0":   1.00,
    "Q5_K_M": 0.65,
    "Q4_K_M": 0.55,
    "Q3_K_M": 0.45,
    "Q2_K":   0.35,
}


# ---------------------------------------------------------------------------
# Full list of models to fetch
#
# Format: (hf_repo_id, display_name, provider, family, use_case, tags)
# Add any repo here and it will be fetched on the next run.
# ---------------------------------------------------------------------------

TARGET_MODELS = [
    # --- Meta LLaMA ---
    ("meta-llama/Llama-3.1-8B-Instruct",           "LLaMA 3.1 8B Instruct",      "Meta",       "LLaMA",     "general",   ["chat"]),
    ("meta-llama/Llama-3.1-70B-Instruct",          "LLaMA 3.1 70B Instruct",     "Meta",       "LLaMA",     "general",   ["chat"]),
    ("meta-llama/Llama-3.1-405B-Instruct",         "LLaMA 3.1 405B Instruct",    "Meta",       "LLaMA",     "general",   ["chat"]),
    ("meta-llama/Llama-3.2-1B-Instruct",           "LLaMA 3.2 1B Instruct",      "Meta",       "LLaMA",     "general",   ["chat"]),
    ("meta-llama/Llama-3.2-3B-Instruct",           "LLaMA 3.2 3B Instruct",      "Meta",       "LLaMA",     "general",   ["chat"]),
    ("meta-llama/Llama-3.2-11B-Vision-Instruct",   "LLaMA 3.2 Vision 11B",       "Meta",       "LLaMA",     "vision",    ["vision", "multimodal"]),
    ("meta-llama/Llama-3.3-70B-Instruct",          "LLaMA 3.3 70B Instruct",     "Meta",       "LLaMA",     "general",   ["chat"]),

    # --- Mistral / Mixtral ---
    ("mistralai/Mistral-7B-Instruct-v0.3",         "Mistral 7B Instruct",        "Mistral AI", "Mistral",   "general",   ["chat"]),
    ("mistralai/Mistral-Nemo-Instruct-2407",       "Mistral Nemo 12B",           "Mistral AI", "Mistral",   "general",   ["chat"]),
    ("mistralai/Mixtral-8x7B-Instruct-v0.1",       "Mixtral 8x7B Instruct",      "Mistral AI", "Mixtral",   "general",   ["chat", "moe"]),
    ("mistralai/Mixtral-8x22B-Instruct-v0.1",      "Mixtral 8x22B Instruct",     "Mistral AI", "Mixtral",   "general",   ["chat", "moe"]),
    ("mistralai/Mistral-Small-Instruct-2409",      "Mistral Small 22B",          "Mistral AI", "Mistral",   "general",   ["chat"]),

    # --- Google Gemma ---
    ("google/gemma-2-2b-it",                       "Gemma 2 2B",                 "Google",     "Gemma",     "general",   ["chat"]),
    ("google/gemma-2-9b-it",                       "Gemma 2 9B",                 "Google",     "Gemma",     "general",   ["chat"]),
    ("google/gemma-2-27b-it",                      "Gemma 2 27B",                "Google",     "Gemma",     "general",   ["chat"]),

    # --- Microsoft Phi ---
    ("microsoft/Phi-3.5-mini-instruct",            "Phi-3.5 Mini 3.8B",          "Microsoft",  "Phi",       "coding",    ["code", "chat"]),
    ("microsoft/Phi-3-medium-128k-instruct",       "Phi-3 Medium 14B",           "Microsoft",  "Phi",       "coding",    ["code", "reasoning"]),
    ("microsoft/phi-4",                            "Phi-4 14B",                  "Microsoft",  "Phi",       "coding",    ["code", "reasoning"]),

    # --- Alibaba Qwen ---
    ("Qwen/Qwen2.5-0.5B-Instruct",                "Qwen2.5 0.5B",               "Alibaba",    "Qwen",      "general",   ["chat", "multilingual"]),
    ("Qwen/Qwen2.5-1.5B-Instruct",                "Qwen2.5 1.5B",               "Alibaba",    "Qwen",      "general",   ["chat", "multilingual"]),
    ("Qwen/Qwen2.5-3B-Instruct",                  "Qwen2.5 3B",                 "Alibaba",    "Qwen",      "general",   ["chat", "multilingual"]),
    ("Qwen/Qwen2.5-7B-Instruct",                  "Qwen2.5 7B",                 "Alibaba",    "Qwen",      "general",   ["chat", "multilingual"]),
    ("Qwen/Qwen2.5-14B-Instruct",                 "Qwen2.5 14B",                "Alibaba",    "Qwen",      "general",   ["chat", "multilingual"]),
    ("Qwen/Qwen2.5-32B-Instruct",                 "Qwen2.5 32B",                "Alibaba",    "Qwen",      "general",   ["chat", "multilingual"]),
    ("Qwen/Qwen2.5-72B-Instruct",                 "Qwen2.5 72B",                "Alibaba",    "Qwen",      "general",   ["chat", "multilingual"]),
    ("Qwen/Qwen2.5-Coder-7B-Instruct",            "Qwen2.5-Coder 7B",           "Alibaba",    "Qwen",      "coding",    ["code"]),
    ("Qwen/Qwen2.5-Coder-32B-Instruct",           "Qwen2.5-Coder 32B",          "Alibaba",    "Qwen",      "coding",    ["code"]),
    ("Qwen/QwQ-32B",                              "QwQ 32B",                    "Alibaba",    "Qwen",      "reasoning", ["reasoning", "math"]),
    ("Qwen/Qwen2-VL-7B-Instruct",                "Qwen2-VL 7B",                "Alibaba",    "Qwen",      "vision",    ["vision", "multimodal"]),

    # --- DeepSeek ---
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1 Distill 1.5B",  "DeepSeek",   "DeepSeek",  "reasoning", ["reasoning", "math"]),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   "DeepSeek-R1 Distill 7B",    "DeepSeek",   "DeepSeek",  "reasoning", ["reasoning", "math"]),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",  "DeepSeek-R1 Distill 14B",   "DeepSeek",   "DeepSeek",  "reasoning", ["reasoning", "math"]),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  "DeepSeek-R1 Distill 32B",   "DeepSeek",   "DeepSeek",  "reasoning", ["reasoning", "math"]),
    ("deepseek-ai/DeepSeek-V3",                   "DeepSeek-V3 671B",           "DeepSeek",   "DeepSeek",  "general",   ["chat", "moe"]),
    ("deepseek-ai/DeepSeek-R1",                   "DeepSeek-R1 671B",           "DeepSeek",   "DeepSeek",  "reasoning", ["reasoning", "moe"]),

    # --- Code models ---
    ("bigcode/starcoder2-7b",                      "StarCoder2 7B",              "BigCode",    "StarCoder", "coding",    ["code"]),
    ("bigcode/starcoder2-15b",                     "StarCoder2 15B",             "BigCode",    "StarCoder", "coding",    ["code"]),
    ("codellama/CodeLlama-7b-Instruct-hf",         "CodeLlama 7B",               "Meta",       "CodeLlama", "coding",    ["code"]),
    ("codellama/CodeLlama-13b-Instruct-hf",        "CodeLlama 13B",              "Meta",       "CodeLlama", "coding",    ["code"]),
    ("codellama/CodeLlama-34b-Instruct-hf",        "CodeLlama 34B",              "Meta",       "CodeLlama", "coding",    ["code"]),

    # --- Embedding models ---
    ("nomic-ai/nomic-embed-text-v1.5",             "nomic-embed-text v1.5",      "Nomic",      "Nomic",     "embedding", ["embedding"]),
    ("mixedbread-ai/mxbai-embed-large-v1",         "mxbai-embed-large v1",       "MixedBread", "BERT",      "embedding", ["embedding"]),
    ("BAAI/bge-m3",                                "BGE-M3",                     "BAAI",       "BGE",       "embedding", ["embedding", "multilingual"]),
]


# ---------------------------------------------------------------------------
# HuggingFace API helpers
# ---------------------------------------------------------------------------

def fetch_model(repo_id, retry=True):
    """
    Fetch model metadata from the HuggingFace API.
    Automatically retries once if rate-limited (HTTP 429).
    """
    url = f"{HF_API_BASE}/models/{repo_id}"
    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "llm-compat-checker-scraper/2.0",
                "Accept":     "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode())

    except urllib.error.HTTPError as e:
        if e.code == 401:
            print(f"  Skipped (private repo): {repo_id}")
        elif e.code == 404:
            print(f"  Skipped (not found): {repo_id}")
        elif e.code == 429 and retry:
            print("  Rate limited — waiting 15 seconds...")
            time.sleep(15)
            return fetch_model(repo_id, retry=False)
        return None
    except Exception as err:
        print(f"  Error fetching {repo_id}: {err}")
        return None


def extract_params(data, repo_id, display_name):
    """
    Determine the parameter count from HuggingFace metadata.
    Tries three approaches in order of reliability.
    """
    # 1. Exact count from safetensors index
    try:
        total = data.get("safetensors", {}).get("total", 0)
        if total and total > 0:
            return round(total / 1e9, 1)
    except Exception:
        pass

    # 2. Size tags like "7b", "70billion"
    for tag in data.get("tags", []):
        t = tag.lower()
        for suffix in ["b", "billion"]:
            if t.endswith(suffix):
                try:
                    return float(t.replace(suffix, "").strip())
                except ValueError:
                    pass

    # 3. Infer from the model name
    combined = (repo_id + display_name).lower().replace(" ", "")
    for size in [405, 141, 70, 72, 67, 34, 32, 27, 22, 14, 13, 12, 11, 9, 8, 7, 4, 3, 2, 1.5, 1.1, 0.5]:
        if f"{size}b" in combined:
            return float(size)

    return None


def build_entry(repo_id, display_name, provider, family, use_case, tags, data):
    """
    Build a model dict from HuggingFace API data.
    Returns None if we can't determine the parameter count.
    """
    params = extract_params(data, repo_id, display_name)
    if params is None:
        return None

    is_moe  = "moe" in tags or any(k in repo_id.lower() for k in ["8x", "moe", "mixture"])
    ram_gb  = int(max(round(params * 2.0 * 1.2), 4))
    vram_gb = int(max(round(params * 2.0 * 1.1), 2))

    # Try to get context window from model card data
    context = 4096
    try:
        card = data.get("cardData") or {}
        for key in ("context_length", "max_position_embeddings", "max_seq_len"):
            val = card.get(key)
            if val:
                context = int(val)
                break
    except Exception:
        pass

    # Assign quantization options based on size
    if params <= 3:
        quants = ["Q4_K_M", "Q5_K_M", "Q8_0"]
    elif params <= 7:
        quants = ["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q8_0"]
    elif params <= 13:
        quants = ["Q3_K_M", "Q4_K_M", "Q5_K_M"]
    elif params <= 34:
        quants = ["Q3_K_M", "Q4_K_M"]
    else:
        quants = ["Q2_K", "Q3_K_M", "Q4_K_M"]

    return {
        "name":      display_name,
        "provider":  provider,
        "family":    family,
        "params":    params,
        "context":   context,
        "ram":       ram_gb,
        "vram":      vram_gb,
        "quants":    quants,
        "use_case":  use_case,
        "tags":      tags,
        "is_moe":    is_moe,
        "hf_repo":   repo_id,
        "likes":     data.get("likes", 0),
        "downloads": data.get("downloads", 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch model metadata from HuggingFace and save to models_db.json"
    )
    parser.add_argument("--limit",  type=int,   default=None, help="Stop after this many models")
    parser.add_argument("--search", type=str,   default=None, help="Only fetch models matching this string")
    parser.add_argument("--delay",  type=float, default=0.25, help="Seconds to wait between requests (default: 0.25)")
    args = parser.parse_args()

    # Apply filters to the target list
    targets = TARGET_MODELS
    if args.search:
        query   = args.search.lower()
        targets = [t for t in targets if query in t[0].lower() or query in t[1].lower()]
    if args.limit:
        targets = targets[:args.limit]

    print(f"\nFetching {len(targets)} models from HuggingFace...\n")

    models        = []
    success_count = 0
    skipped_count = 0

    for i, (repo_id, display_name, provider, family, use_case, tags) in enumerate(targets, 1):
        # Pad the index and name for aligned output
        prefix = f"  [{i:>2}/{len(targets)}]  {display_name:<50}"
        print(prefix, end="", flush=True)

        data = fetch_model(repo_id)
        if data:
            entry = build_entry(repo_id, display_name, provider, family, use_case, tags, data)
            if entry:
                models.append(entry)
                print(f"  {entry['params']}B  /  {entry['ram']} GB RAM  /  ctx {entry['context'] // 1000}K")
                success_count += 1
            else:
                print("  (could not extract parameter count, skipped)")
                skipped_count += 1
        else:
            print("  (fetch failed, skipped)")
            skipped_count += 1

        time.sleep(args.delay)

    # Sort by parameter count before saving
    models.sort(key=lambda m: m["params"])

    db = {
        "updated_at":  datetime.now().isoformat(),
        "model_count": len(models),
        "source":      "huggingface_api",
        "models":      models,
    }
    OUTPUT_PATH.write_text(json.dumps(db, indent=2))

    print(f"\n  {success_count} models fetched successfully")
    if skipped_count:
        print(f"  {skipped_count} models skipped")
    print(f"  {len(models)} total models saved to {OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
