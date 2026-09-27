"""
TURKLM - Pretraining Data Preparation & Packing Pipeline
Cleans, deduplicates, tokenizes, and packs Turkish text into fixed context length blocks (e.g., 2048).
Generates the exact cache structure and `_COMPLETE` validation marker required by `train_base.py`.
"""

import os
import argparse
import hashlib
from typing import Dict, Any, List
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm


def prepare_pretraining_corpus(
    tokenizer_path: str,
    output_cache_dir: str,
    context_length: int = 2048,
    dataset_name: str = "wikipedia",
    subset: str = "20220301.tr",
    eval_ratio: float = 0.03,
):
    print("=" * 80)
    print("TURKLM PRETRAINING DATA PREPARATION & PACKING")
    print("=" * 80)

    # 1. Load Tokenizer & Calculate Fingerprint
    print(f"[*] Loading tokenizer from {tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    with open(os.path.join(tokenizer_path, "tokenizer.json"), "rb") as f:
        tok_hash = hashlib.sha256(f.read()).hexdigest()[:12]
    print(f"[+] Tokenizer loaded: vocab_size={len(tokenizer)}, SHA-256={tok_hash}")

    # 2. Download / Load Raw Data
    print(f"[*] Loading raw dataset: {dataset_name} ({subset})...")
    raw_ds = load_dataset(dataset_name, subset, split="train")
    print(f"[+] Total raw documents: {len(raw_ds):,}")

    # 3. Clean & Deduplicate (Document-level)
    seen_hashes = set()
    unique_texts = []
    print("[*] Deduplicating documents via SHA-256 content hashing...")
    for text in tqdm(raw_ds["text"], desc="Deduplicating"):
        cleaned = text.strip()
        if len(cleaned) < 50:  # Skip trivial texts
            continue
        content_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_texts.append(cleaned)

    print(f"[+] Unique high-quality documents: {len(unique_texts):,}")

    # 4. Tokenization & Packing into Fixed Context Windows
    print(f"[*] Packing tokens into fixed context blocks (block_size={context_length})...")
    bos_id = tokenizer.bos_token_id or 0
    eos_id = tokenizer.eos_token_id or 2

    all_blocks = []
    current_buffer = []

    for text in tqdm(unique_texts, desc="Tokenizing & Packing"):
        tokens = [bos_id] + tokenizer.encode(text, add_special_tokens=False) + [eos_id]
        current_buffer.extend(tokens)

        while len(current_buffer) >= context_length:
            block = current_buffer[:context_length]
            all_blocks.append(block)
            current_buffer = current_buffer[context_length:]

    print(f"[+] Total packed training blocks generated: {len(all_blocks):,}")
    print(f"[+] Total tokens in corpus: {len(all_blocks) * context_length:,}")

    # 5. Train / Eval Split
    total_samples = len(all_blocks)
    eval_count = max(100, int(total_samples * eval_ratio))
    train_count = total_samples - eval_count

    train_data = {"input_ids": all_blocks[:train_count]}
    eval_data = {"input_ids": all_blocks[train_count:]}

    from datasets import Dataset
    train_ds = Dataset.from_dict(train_data)
    eval_ds = Dataset.from_dict(eval_data)
    dataset_dict = DatasetDict({"train": train_ds, "eval": eval_ds})

    # 6. Save to disk with _COMPLETE marker
    os.makedirs(output_cache_dir, exist_ok=True)
    print(f"[*] Saving packed dataset to {output_cache_dir}...")
    dataset_dict.save_to_disk(output_cache_dir)

    complete_marker_path = os.path.join(output_cache_dir, "_COMPLETE")
    with open(complete_marker_path, "w", encoding="utf-8") as f:
        f.write("OK\n")

    print(f"[✓] Data preparation complete. Marker written: {complete_marker_path}")
    print(f"    - Train blocks: {train_count:,}")
    print(f"    - Eval blocks : {eval_count:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and Pack Pretraining Data for TURKLM")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to trained tokenizer directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for packed dataset cache")
    parser.add_argument("--context_len", type=int, default=2048, help="Context block size")
    args = parser.parse_args()

    prepare_pretraining_corpus(
        tokenizer_path=args.tokenizer_path,
        output_cache_dir=args.output_dir,
        context_length=args.context_len,
    )
