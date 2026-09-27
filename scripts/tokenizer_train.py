"""
TURKLM - Custom Turkish BPE Tokenizer Training Script
Trains a Byte-Level Byte-Pair Encoding (BBPE) tokenizer with Turkish morphological support.
"""

import os
import argparse
from typing import List, Generator
from datasets import load_dataset
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS = ["<unk>", "<pad>", "<s>", "</s>", "<mask>"]
VOCAB_SIZE_DEFAULT = 64000


def get_training_corpus(dataset_name: str, split: str = "train", text_column: str = "text",
                        batch_size: int = 1000) -> Generator[List[str], None, None]:
    """Stream dataset in batches to minimize memory overhead during tokenizer training."""
    ds = load_dataset(dataset_name, split=split, streaming=True)
    batch = []
    for example in ds:
        text = example.get(text_column, "")
        if text and len(text.strip()) > 10:
            batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def train_turkish_tokenizer(output_dir: str, vocab_size: int = VOCAB_SIZE_DEFAULT,
                            dataset_name: str = "wikipedia", subset: str = "20220301.tr"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Training Turkish Byte-Level BPE Tokenizer (vocab_size={vocab_size})...")

    # 1. Initialize empty BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # 2. Normalization & Pre-tokenization
    # NFC normalization handles Turkish diacritics (ç, ğ, ı, ö, ş, ü) gracefully
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=False)
    ])
    tokenizer.decoder = decoders.ByteLevel()

    # 3. Setup BPE Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    # 4. Stream and Train
    print(f"[*] Fetching corpus from {dataset_name} ({subset})...")
    ds = load_dataset(dataset_name, subset, split="train", streaming=True)
    
    def batch_iterator():
        batch = []
        for row in ds:
            text = row.get("text", "")
            if text:
                batch.append(text)
            if len(batch) == 1000:
                yield batch
                batch = []
        if batch:
            yield batch

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # 5. Post-Processing (BOS / EOS template)
    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[
            ("<s>", bos_id),
            ("</s>", eos_id),
        ],
    )

    # 6. Save native tokenizers JSON
    raw_json_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(raw_json_path)
    print(f"[+] Saved raw tokenizer to {raw_json_path}")

    # 7. Wrap and export as Hugging Face PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=raw_json_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    hf_tokenizer.save_pretrained(output_dir)
    print(f"[+] Exported Hugging Face tokenizer artifact to {output_dir}")
    print(f"[+] Total vocabulary count: {len(hf_tokenizer)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Turkish BPE Tokenizer")
    parser.add_argument("--output_dir", type=str, default="./tokenizer", help="Directory to save tokenizer")
    parser.add_argument("--vocab_size", type=int, default=64000, help="Vocabulary size")
    args = parser.parse_args()

    train_turkish_tokenizer(args.output_dir, args.vocab_size)
