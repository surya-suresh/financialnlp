import argparse
import json
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_pairs(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("date", ""))
    return rows


def chrono_split(rows: list[dict], train_ratio: float = 0.9):
    n = len(rows)
    k = int(n * train_ratio)
    return rows[:k], rows[k:]


def balance_classes(rows: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = {}
    for r in rows:
        by_class.setdefault(r["output"], []).append(r)
    n_per_class = min(len(v) for v in by_class.values())
    out = []
    for examples in by_class.values():
        out.extend(rng.sample(examples, n_per_class))
    rng.shuffle(out)
    return out


def head_tail_truncate(tokenizer, text: str, max_length: int,
                       head_ratio: float = 0.6) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_length:
        return ids
    head_n = int(max_length * head_ratio)
    tail_n = max_length - head_n
    return ids[:head_n] + ids[-tail_n:]


def oversample_minority(rows: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = {}
    for r in rows:
        by_class.setdefault(r["output"], []).append(r)
    max_count = max(len(v) for v in by_class.values())
    out = []
    for examples in by_class.values():
        out.extend(examples)
        shortfall = max_count - len(examples)
        if shortfall > 0:
            out.extend(rng.choices(examples, k=shortfall))
    rng.shuffle(out)
    return out


class PairDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt = row["input"] + "\n\nAnswer:"
        answer = " " + row["output"] + self.tokenizer.eos_token

        ans_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        prompt_max = max(self.max_length - len(ans_ids), 32)
        prompt_ids = head_tail_truncate(self.tokenizer, prompt, prompt_max)

        input_ids = prompt_ids + ans_ids
        labels = [-100] * len(prompt_ids) + list(ans_ids)
        attention = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(batch, pad_id: int):
    max_len = max(len(b["input_ids"]) for b in batch)

    def _pad(tensors, value):
        out = torch.full((len(tensors), max_len), value, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            out[i, :len(t)] = t
        return out

    return {
        "input_ids": _pad([b["input_ids"] for b in batch], pad_id),
        "attention_mask": _pad([b["attention_mask"] for b in batch], 0),
        "labels": _pad([b["labels"] for b in batch], -100),
    }


def build_model(base_model: str, bf16: bool = False):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balance-test", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer, Trainer, TrainingArguments

    rows = load_pairs(args.pairs)
    logger.info("Loaded %d pairs from %s", len(rows), args.pairs)

    train, val = chrono_split(rows, 0.9)
    logger.info("Train=%d  Val=%d (before balancing)", len(train), len(val))

    counts = {}
    for r in rows:
        counts[r["output"]] = counts.get(r["output"], 0) + 1
    logger.info("Class balance (all):   %s", counts)
    logger.info("Class balance (train): %s", dict(Counter(r["output"] for r in train)))

    train = oversample_minority(train, seed=args.seed)
    logger.info("Class balance (train, oversampled): %s  n=%d",
                dict(Counter(r["output"] for r in train)), len(train))

    if args.balance_test:
        val = balance_classes(val, seed=args.seed)
        logger.info("Val balanced: %s  n=%d",
                    dict(Counter(r["output"] for r in val)), len(val))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(args.base_model, bf16=args.bf16)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    train_ds = PairDataset(train, tokenizer, args.max_length)
    val_ds = PairDataset(val, tokenizer, args.max_length)

    args.out.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=not args.bf16,
        bf16=args.bf16,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
    )
    trainer.train()

    adapter_dir = args.out / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Saved adapter to %s", adapter_dir)


if __name__ == "__main__":
    main()
