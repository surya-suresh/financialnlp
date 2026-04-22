"""Smoke test: exercise the exact code path that was failing at job start.

Runs on a GPU node via salloc. If this prints 'SMOKE OK', the real jobs will
get past the model-loading step.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
print("torch:", torch.__version__, "cuda avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0),
          "capability:", torch.cuda.get_device_capability(0))

import bitsandbytes as bnb
import bitsandbytes.functional as F
print("bnb:", bnb.__version__,
      "has cquantize_blockwise_fp16_nf4:",
      hasattr(F.lib, "cquantize_blockwise_fp16_nf4"))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE = "Qwen/Qwen2.5-7B-Instruct"
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("loading tokenizer ...")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("loading model 4-bit ...")
model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb_cfg, device_map="auto", trust_remote_code=True,
)
print("model device map:", {n: str(p.device) for n, p in list(model.named_parameters())[:2]})

model = prepare_model_for_kbit_training(model)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()

print("running one forward pass ...")
prompt = "Answer in one word: up or down?\n\nAnswer:"
ids = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model(**ids)
print("logits shape:", out.logits.shape)

print("SMOKE OK")
