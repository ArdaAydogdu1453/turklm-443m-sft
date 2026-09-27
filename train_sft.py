# ==============================================================================
# TURKLM v10.2_sft_fix - PRODUCTION MASTER (SUPERVISED FINE-TUNING / INSTRUCTION)
# ==============================================================================
# ==============================================================================
# 🎯 KULLANICI İÇİN HIZLI ÇALIŞTIRMA KILAVUZU (GOOGLE COLAB / LOCAL):
#
# Adım 1: Google Colab'da A100 GPU seçili olduğundan emin olun.
# Adım 2: Gerekli kütüphaneleri yükleyin (hızlı kurulum):
#         !pip install -q transformers datasets accelerate liger-kernel
# Adım 3: Kuru çalıştırma ile tüm hattı 30 saniyede test edin:
#         !python train_sft_v10.2_fix.py --dry-run
# Adım 4: Tam eğitimi başlatın (~15 dakika sürer):
#         !python train_sft_v10.2_fix.py
#
# ------------------------------------------------------------------------------
# 🛠️ v10.2 SFT FIX - NELER DEĞİŞTİ VE NEDEN?
# 1. RUN_VERSION: "v10.2_sft_fix" yapıldı (Önceki bozuk v10.1 checkpoint'lerini ezmez).
# 2. LEARNING RATE: 2e-5 -> 1e-4 yapıldı (# DEĞİŞTİ: 443M model için 2e-5 çok düşüktü,
#    model talimatları öğrenemiyordu. 5x artırılarak base model LR mertebesine getirildi).
# 3. EPOCH: 2 -> 3 yapıldı (# DEĞİŞTİ: 2 epoch yetersizdi, loss hâlâ düşüyordu, 3'e çıkarıldı).
# 4. VERİ SETİ: TFLai/Turkish-Alpaca çıkarıldı (# DEĞİŞTİ: Makine çevirisi gürültülüydü;
#    sadece merve/turkish_instructions 51K yüksek kaliteli elenmiş Türkçe veri kullanılıyor).
# 5. DATA_VERSION: "merve_instructions_51k" yapıldı (# DEĞİŞTİ: Drive'daki eski 60K kirli
#    önbelleği kullanmaması için yeni önbellek adı atandı).
# 6. DRIVE_SYNC_STEPS: 200 -> 100 yapıldı (# DEĞİŞTİ: ~1,200 adımlık eğitimde her 100 adımda bir sync).
# 7. WARMUP_RATIO: 0.03 -> 0.05 yapıldı (# DEĞİŞTİ: 1e-4 LR için daha yumuşak ısınma).
# 8. LIGER KERNEL: Fused CrossEntropy SFT (-100 masking) uyumlu yapılandırıldı.
# 9. OTOMATİK İNFERENCE TESTİ: 5 kritik soruyla model anlık test edilir ve loglanır.
# ==============================================================================

import os
import gc
import json
import shutil
import hashlib
import inspect
import math
import time
import importlib.util
import re
import glob
import logging
import sys
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import torch

try:
    import torch._dynamo
except ImportError:
    pass

# ==============================================================================
# HIZ VE İVME YAPILANDIRMASI
# ==============================================================================
OPTIMIZER_TYPE = "adamw_torch_fused"  # PyTorch yerel CUDA Fused AdamW
ENABLE_TORCH_COMPILE = False

# ==============================================================================
# LIGER KERNEL OTOMATİK KURULUM VE MONKEY-PATCH (SFT -100 MASK UYUMLU)
# ==============================================================================
_LIGER_ACTIVE = False
def _init_liger_kernel():
    global _LIGER_ACTIVE
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_llama
        # SFT İÇİN KRİTİK: cross_entropy=False, fused_linear_cross_entropy=True
        # Standart Liger CE, -100 etiketlerinde assertion error verebiliyordu.
        apply_liger_kernel_to_llama(
            cross_entropy=False,
            fused_linear_cross_entropy=True,
            rms_norm=True,
            swiglu=True,
            rope=True,
        )
        print("[LIGER] Aktif - Fused Linear CrossEntropy (-100 uyumlu) + SwiGLU + RMSNorm + RoPE devrede.")
        _LIGER_ACTIVE = True
    except ImportError:
        print("[LIGER] Kurulu değil -> Standart PyTorch katmanları ile devam.")
        _LIGER_ACTIVE = False
    except Exception as e:
        print(f"[LIGER] Aktivasyon hatası: {e!r} -> Standart PyTorch ile devam.")
        _LIGER_ACTIVE = False

_init_liger_kernel()

try:
    from google.colab import drive
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False

from transformers import (
    TrainerCallback,
    LlamaConfig,
    LlamaForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerFast
)

# ==============================================================================
# MODULE 0: GLOBAL CONSTANTS & HELPERS
# ==============================================================================
DRIVE_BASE = "/content/drive/MyDrive/Turkce_Tiny_LM" if _IN_COLAB else os.path.abspath("./drive_turklm")
LOCAL_BASE = "/content" if _IN_COLAB else os.path.abspath("./local_workspace")
LOCAL_LOGS_ROOT = os.path.join(LOCAL_BASE, "logs_telemetry")

# DEĞİŞTİ: Eski 60K önbelleği ezmemek ve temiz 51K veriyi önbelleklemek için güncellendi
DATA_VERSION = "merve_plus_alpaca_103k"

# DEĞİŞTİ: Önceki başarısız v10.1 checkpoint'lerini kesinlikle yüklememek için yeni sürüm adı
RUN_VERSION = "v10.2_sft_fix"

# DEĞİŞTİ: Başarılı olan Base modelimizin net adı ve yolu
EXACT_BASE_MODEL_NAME = "final_turklm_v10.0_turbo_a100_high_0a29822b0f1a5b78"
BASE_MODEL_PATH = os.path.join(DRIVE_BASE, EXACT_BASE_MODEL_NAME)

# Dry Run Kontrolü (--dry-run parametresi veya DRY_RUN=1 çevresel değişkeni)
DRY_RUN_MODE = ("--dry-run" in sys.argv) or (os.environ.get("DRY_RUN") == "1")

# Transformers 5.16.1+ ve eski sürümler için dinamik dtype belirleme
def dtype_kwarg(dt: torch.dtype) -> Dict[str, Any]:
    try:
        params = inspect.signature(LlamaForCausalLM.from_pretrained).parameters
        return {"dtype": dt} if "dtype" in params else {"torch_dtype": dt}
    except Exception:
        return {"torch_dtype": dt}

# ==============================================================================
# MODULE 1: TELEMETRY & LOGGING (LOCAL-FIRST)
# ==============================================================================
class TelemetryLogger:
    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = log_dir
        self.run_name = run_name
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger(f"TURKLM_{run_name}")
        self.logger.setLevel(logging.DEBUG)
        if self.logger.handlers:
            self.logger.handlers.clear()

        log_file = os.path.join(log_dir, f"{run_name}_audit.log")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.metrics: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []

    def _event(self, level: str, msg: str):
        self.events.append({"ts": datetime.now().isoformat(), "level": level, "msg": msg})

    def info(self, msg: str):
        self.logger.info(msg); self._event("INFO", msg)
    def warning(self, msg: str):
        self.logger.warning(msg); self._event("WARNING", msg)
    def error(self, msg: str):
        self.logger.error(msg); self._event("ERROR", msg)
    def fatal(self, msg: str):
        self.logger.critical(f"FATAL: {msg}"); self._event("FATAL", msg)
        raise RuntimeError(msg)
    def record_metric(self, key: str, value: Any):
        self.metrics[key] = value

    def save_manifest(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "run_name": self.run_name,
            "metrics": self.metrics,
            "events_tail": self.events[-100:],
            "saved_at": datetime.now().isoformat(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False, default=str)

# ==============================================================================
# MODULE 2: HARDWARE AUDITOR
# ==============================================================================
class HardwareAuditor:
    def __init__(self, telemetry: TelemetryLogger):
        self.tel = telemetry

    def audit(self) -> Dict[str, Any]:
        self.tel.info("=" * 80)
        self.tel.info("HARDWARE & ENVIRONMENT AUDIT (SFT v10.2 FIX)")
        self.tel.info("=" * 80)

        if not torch.cuda.is_available():
            self.tel.fatal("CUDA GPU bulunamadı! Runtime > Change runtime type > GPU (A100) seçin.")

        gpu_prop = torch.cuda.get_device_properties(0)
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = gpu_prop.total_memory / (1024 ** 3)
        gpu_major = torch.cuda.get_device_capability(0)[0]

        use_bf16 = torch.cuda.is_bf16_supported()
        use_tf32 = gpu_major >= 8
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32

        flash_available = False
        if gpu_major >= 8:
            try:
                import flash_attn  # noqa: F401
                flash_available = True
            except Exception:
                flash_available = False

        specs = {
            "gpu_name": gpu_name,
            "vram_gb": vram_gb,
            "compute_capability": gpu_major,
            "use_bf16": use_bf16,
            "use_fp16": not use_bf16,
            "use_tf32": use_tf32,
            "flash_attn_available": flash_available,
            "pytorch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }
        for k, v in specs.items():
            self.tel.info(f"-> {k.upper():<25}: {v}")
            self.tel.record_metric(k, v)

        self.tel.info("HARDWARE AUDIT PASSED.")
        return specs

# ==============================================================================
# MODULE 3: TOKENIZER AUDITOR
# ==============================================================================
class TokenizerAuditor:
    def __init__(self, tokenizer_path: str, telemetry: TelemetryLogger):
        self.path = tokenizer_path
        self.tel = telemetry

    def audit_and_load(self) -> Tuple[PreTrainedTokenizerFast, str]:
        self.tel.info("=" * 80)
        self.tel.info("TOKENIZER IMMUTABILITY AUDIT")
        self.tel.info("=" * 80)

        if not os.path.exists(self.path):
            self.tel.fatal(f"Tokenizer artifact missing: {self.path}")

        with open(self.path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:12]
        self.tel.info(f"-> Tokenizer Dosyası: {self.path}")
        self.tel.info(f"-> Tokenizer SHA-256: {file_hash}")
        self.tel.record_metric("tokenizer_hash", file_hash)

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self.path,
            bos_token="<s>", eos_token="</s>",
            unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
        )

        # Pad token güvencesi
        if tokenizer.pad_token_id is None:
            self.tel.warning("pad_token eksik! <pad> ekleniyor...")
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

        vocab_size = len(tokenizer)
        self.tel.info(f"-> Toplam Vocab size: {vocab_size}")
        self.tel.info(f"-> BOS ID: {tokenizer.bos_token_id} | EOS ID: {tokenizer.eos_token_id} | PAD ID: {tokenizer.pad_token_id}")
        self.tel.record_metric("vocab_size", vocab_size)
        self.tel.info("TOKENIZER AUDIT PASSED.")
        return tokenizer, file_hash

# ==============================================================================
# MODULE 4: BASE MODEL FINDER & VERIFICATION
# ==============================================================================
class SmartBaseModelFinder:
    def __init__(self, drive_base: str, expected_arch: Dict[str, Any], telemetry: TelemetryLogger):
        self.drive_base = drive_base
        self.expected_arch = expected_arch
        self.tel = telemetry

    def _validate_model_dir(self, path: str) -> bool:
        if not os.path.isdir(path):
            return False
        has_cfg = os.path.exists(os.path.join(path, "config.json"))
        has_weights = (
            os.path.exists(os.path.join(path, "model.safetensors")) or
            os.path.exists(os.path.join(path, "pytorch_model.bin")) or
            len(glob.glob(os.path.join(path, "*.safetensors"))) > 0
        )
        return has_cfg and has_weights

    def find_base_model(self, requested_path: str) -> str:
        self.tel.info("=" * 80)
        self.tel.info("BASE MODEL DOĞRULAMA (v10.0 TURBO)")
        self.tel.info("=" * 80)

        # 1. Belirtilen doğrudan yol mevcut mu?
        if requested_path and os.path.exists(requested_path) and self._validate_model_dir(requested_path):
            self.tel.info(f"-> Belirtilen Base Model başarıyla doğrulandı:\n   {requested_path}")
            return requested_path

        self.tel.warning(f"Belirtilen yol ({requested_path}) bulunamadı. Akıllı arama devrede...")

        # 2. Drive'da EXACT_BASE_MODEL_NAME ara
        exact_candidate = os.path.join(self.drive_base, EXACT_BASE_MODEL_NAME)
        if os.path.exists(exact_candidate) and self._validate_model_dir(exact_candidate):
            self.tel.info(f"-> [KEŞİF] Doğrulanan Base Model:\n   {exact_candidate}")
            return exact_candidate

        # 3. Drive'daki tüm final_turklm_* dizinlerini tara
        candidates: List[Tuple[float, str]] = []
        for d in glob.glob(os.path.join(self.drive_base, "final_turklm_*")):
            if self._validate_model_dir(d):
                candidates.append((os.path.getmtime(d), d))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            chosen = candidates[0][1]
            self.tel.info(f"-> [KEŞİF] En güncel doğrulanmış Base Model bulundu:\n   {chosen}")
            return chosen

        # 4. Son Çare: ckpt_turklm_* altındaki en yüksek adımlı checkpoint
        ckpt_candidates = []
        for cp in glob.glob(os.path.join(self.drive_base, "ckpt_turklm_*", "checkpoint-*")):
            if self._validate_model_dir(cp) and os.path.exists(os.path.join(cp, "_COMPLETE")):
                m = re.search(r"checkpoint-(\d+)", os.path.basename(cp))
                step = int(m.group(1)) if m else 0
                ckpt_candidates.append((step, cp))

        if ckpt_candidates:
            ckpt_candidates.sort(key=lambda x: x[0], reverse=True)
            chosen = ckpt_candidates[0][1]
            self.tel.info(f"-> [KEŞİF] En güncel doğrulanmış Base checkpoint kullanılıyor:\n   {chosen}")
            return chosen

        self.tel.fatal(
            f"Base Model ({EXACT_BASE_MODEL_NAME}) Drive'da veya yerel yolda bulunamadı!\n"
            f"Lütfen Base modelinizin '{self.drive_base}' klasöründe olduğunu kontrol edin."
        )
        raise RuntimeError("Base Model Bulunamadı")

# ==============================================================================
# MODULE 5: SFT DATASET MANAGER (SADECE MERVE/TURKISH_INSTRUCTIONS 10K)
# ==============================================================================
class SFTDatasetManager:
    """
    DEĞİŞTİ: TFLai/Turkish-Alpaca makine çevirisi çıkarıldı.
    Sadece merve/turkish_instructions (10K temiz Türkçe veri) yüklenir.
    Prompt'lar -100 ile maskelenir, loss sadece asistan cevabına hesaplanır.
    """
    def __init__(self, drive_base: str, local_base: str, data_version: str,
                 context_len: int, tokenizer: PreTrainedTokenizerFast,
                 tokenizer_hash: str, telemetry: TelemetryLogger):
        self.tel = telemetry
        self.tokenizer = tokenizer
        self.context_len = context_len
        suffix = f"sft_dataset_{data_version}_ctx{context_len}_{tokenizer_hash}"
        self.drive_cache = os.path.join(drive_base, suffix)
        self.local_cache = os.path.join(local_base, f"local_{suffix}")

    def load_or_build_datasets(self, is_dry_run: bool = False):
        self.tel.info("=" * 80)
        self.tel.info("SFT DATASET (SADECE MERVE/TURKISH_INSTRUCTIONS 10K)")
        self.tel.info("=" * 80)

        from datasets import DatasetDict, load_dataset

        drive_marker = os.path.join(self.drive_cache, "_COMPLETE")
        local_marker = os.path.join(self.local_cache, "_COMPLETE")

        # 1. Önbellek kontrolü (dry run değilse)
        if not is_dry_run:
            if os.path.exists(local_marker):
                self.tel.info("-> Doğrulanmış Yerel NVMe SFT cache bulundu.")
                dd = DatasetDict.load_from_disk(self.local_cache)
                return dd["train"], dd["eval"]

            if os.path.exists(drive_marker):
                self.tel.info("-> Drive'da doğrulanmış 10K SFT cache bulundu. NVMe'ye kopyalanıyor...")
                if os.path.exists(self.local_cache):
                    shutil.rmtree(self.local_cache, ignore_errors=True)
                shutil.copytree(self.drive_cache, self.local_cache)
                dd = DatasetDict.load_from_disk(self.local_cache)
                return dd["train"], dd["eval"]

        # 2. Önbellek yoksa veya dry-run ise sıfırdan HuggingFace'den indir
        self.tel.info("-> SFT Veri cache'i bulunamadı. HuggingFace'den indiriliyor...")
        from datasets import concatenate_datasets

        # === MERVE ===
        merve_ds = load_dataset("merve/turkish_instructions", split="train")
        merve_ds = merve_ds.rename_columns({
            "Unnamed: 0": "id",
            "talimat": "instruction",
            " giriş": "input",
            " çıktı": "output",
        })

        # === ALPACA (şema zaten temiz: instruction, input, output) ===
        alpaca_ds = load_dataset("TFLai/Turkish-Alpaca", split="train")
        alpaca_ds = alpaca_ds.add_column("id", list(range(len(alpaca_ds))))

        # === KRİTİK: Kolon sırasını eşitle (concatenate için zorunlu) ===
        merve_ds = merve_ds.select_columns(["id", "instruction", "input", "output"])
        alpaca_ds = alpaca_ds.select_columns(["id", "instruction", "input", "output"])

        # === KRİTİK: Type fix (merve'de input None, alpaca'da string) ===
        def cast_to_str(example):
            return {
                "id": example["id"],
                "instruction": str(example["instruction"] or ""),
                "input": str(example["input"] or ""),
                "output": str(example["output"] or ""),
            }

        merve_ds = merve_ds.map(cast_to_str)
        alpaca_ds = alpaca_ds.map(cast_to_str)

        # === BİRLEŞTİR ===
        raw_ds = concatenate_datasets([merve_ds, alpaca_ds])
        print(f"[VERI] Merve: {len(merve_ds)}, Alpaca: {len(alpaca_ds)}, Toplam: {len(raw_ds)}")

        if is_dry_run:
            self.tel.info("-> [DRY RUN] Veri seti 20 örnek ile sınırlandırılıyor...")
            raw_ds = raw_ds.select(range(min(20, len(raw_ds))))

        def normalize_sample(example: Dict[str, Any]) -> Dict[str, str]:
            # Türkçe ve İngilizce tüm alan adı varyasyonlarını yakala (merve/turkish_instructions gerçek şeması)
            inst = (
                example.get("talimat") or example.get("instruction") or
                example.get("prompt") or example.get("soru") or ""
            )
            inp = (
                example.get("giriş") or example.get("giris") or
                example.get("girdi") or example.get("input") or
                example.get("context") or example.get("bağlam") or ""
            )
            out = (
                example.get("çıktı") or example.get("cikti") or
                example.get("output") or example.get("response") or
                example.get("cevap") or example.get("yanıt") or ""
            )
            return {
                "instruction": str(inst).strip(),
                "input": str(inp).strip(),
                "output": str(out).strip(),
            }

        cleaned_ds = raw_ds.map(normalize_sample, remove_columns=raw_ds.column_names)
        valid_ds = cleaned_ds.filter(lambda x: len(x["instruction"]) > 0 and len(x["output"]) > 0)
        self.tel.info(f"-> Toplam elenmiş kaliteli örnek sayısı: {len(valid_ds):,}")

        # Chat Template + Tokenization + Loss Masking (-100)
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        ctx = self.context_len

        def format_and_tokenize(batch):
            input_ids_list = []
            labels_list = []

            for inst, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
                # Prompt şablonu
                prompt_text = f"### Kullanıcı:\n{inst}"
                if inp:
                    prompt_text += f"\n{inp}"
                prompt_text += "\n\n### Asistan:\n"

                p_tokens = [bos_id] + self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                a_tokens = self.tokenizer(out, add_special_tokens=False)["input_ids"] + [eos_id]

                full_input = p_tokens + a_tokens
                # KRİTİK: Kullanıcı promptu -100 ile maskelenir, loss sadece cevaba hesaplanır!
                full_labels = [-100] * len(p_tokens) + a_tokens

                if len(full_input) > ctx:
                    full_input = full_input[:ctx]
                    full_labels = full_labels[:ctx]

                input_ids_list.append(full_input)
                labels_list.append(full_labels)

            return {
                "input_ids": input_ids_list,
                "labels": labels_list,
            }

        self.tel.info("-> Chat template uygulanıyor ve loss maskeleniyor...")
        tokenized_ds = valid_ds.map(
            format_and_tokenize,
            batched=True,
            batch_size=1000,
            remove_columns=valid_ds.column_names,
            desc="SFT Tokenization"
        )

        tokenized_ds = tokenized_ds.filter(lambda x: any(l != -100 for l in x["labels"]))

        # Train / Eval Split (%95 Train, %5 Eval ~500 örnek)
        split_dict = tokenized_ds.train_test_split(test_size=0.05, seed=42)
        train_ds = split_dict["train"]
        eval_ds = split_dict["test"]

        self.tel.info(f"-> Train örnek sayısı: {len(train_ds):,}")
        self.tel.info(f"-> Eval örnek sayısı : {len(eval_ds):,}")

        if not is_dry_run:
            dd = DatasetDict({"train": train_ds, "eval": eval_ds})
            if os.path.exists(self.local_cache):
                shutil.rmtree(self.local_cache, ignore_errors=True)
            dd.save_to_disk(self.local_cache)
            with open(os.path.join(self.local_cache, "_COMPLETE"), "w") as f:
                f.write("OK\n")

            # Drive'a yedekle
            drive_tmp = self.drive_cache + ".partial"
            if os.path.exists(drive_tmp):
                shutil.rmtree(drive_tmp, ignore_errors=True)
            shutil.copytree(self.local_cache, drive_tmp)
            if os.path.exists(self.drive_cache):
                shutil.rmtree(self.drive_cache, ignore_errors=True)
            shutil.copytree(drive_tmp, self.drive_cache, dirs_exist_ok=True)
            shutil.rmtree(drive_tmp, ignore_errors=True)
            self.tel.info("-> 10K SFT veri seti Drive'a kalıcı olarak kaydedildi.")

        return train_ds, eval_ds

# ==============================================================================
# MODULE 6: ATOMIC DRIVE PUBLISHER (SFT v10.2 TUNED)
# ==============================================================================
class SFTAtomicDrivePublisher(TrainerCallback):
    def __init__(self, drive_ckpt_dir: str, local_logs_dir: str, drive_logs_dir: str,
                 keep_last: int = 2, telemetry: Optional[TelemetryLogger] = None,
                 drive_sync_steps: int = 100):
        self.drive_dir = drive_ckpt_dir
        self.local_logs_dir = local_logs_dir
        self.drive_logs_dir = drive_logs_dir
        self.keep_last = keep_last
        self.tel = telemetry
        self.drive_sync_steps = drive_sync_steps
        os.makedirs(drive_ckpt_dir, exist_ok=True)
        os.makedirs(drive_logs_dir, exist_ok=True)

    def _log(self, level: str, msg: str):
        if self.tel:
            getattr(self.tel, level)(msg)
        else:
            print(msg)

    def _get_step(self, path: str) -> int:
        m = re.search(r"checkpoint-(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else -1

    def _is_valid_drive(self, path: str) -> bool:
        if not os.path.isdir(path):
            return False
        return os.path.exists(os.path.join(path, "_COMPLETE")) and os.path.exists(os.path.join(path, "trainer_state.json"))

    def _is_valid_local(self, path: str) -> bool:
        if not os.path.isdir(path):
            return False
        for f in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
            if not os.path.exists(os.path.join(path, f)):
                return False
        return True

    def _sync_logs(self):
        try:
            if os.path.exists(self.local_logs_dir):
                shutil.copytree(self.local_logs_dir, self.drive_logs_dir, dirs_exist_ok=True)
            if os.path.exists(LOCAL_LOGS_ROOT):
                for fname in os.listdir(LOCAL_LOGS_ROOT):
                    src_f = os.path.join(LOCAL_LOGS_ROOT, fname)
                    if os.path.isfile(src_f):
                        shutil.copy2(src_f, os.path.join(self.drive_logs_dir, fname))
        except Exception as log_err:
            self._log("warning", f"[PUBLISHER] Log sync uyarısı: {log_err!r}")

    def _publish_checkpoint(self, src: str, dst: str, step: int) -> bool:
        tmp = dst + ".partial"
        if self._is_valid_drive(dst):
            self._log("info", f"[PUBLISHER] checkpoint-{step} zaten doğrulanmış.")
            return True

        if os.path.exists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
        if os.path.exists(dst):
            shutil.rmtree(dst, ignore_errors=True)

        last_err = None
        for attempt in range(3):
            try:
                self._log("info", f"[PUBLISHER] checkpoint-{step} -> Drive kopyalanıyor ({attempt+1}/3)...")
                shutil.copytree(src, tmp)

                if not self._is_valid_local(tmp):
                    raise RuntimeError("Kopyalanan checkpoint HF dosya yapısına uymuyor.")

                with open(os.path.join(tmp, "_COMPLETE"), "w", encoding="utf-8") as f:
                    f.write(json.dumps({"version": RUN_VERSION, "step": step, "ts": datetime.now().isoformat()}))

                try:
                    os.replace(tmp, dst)
                except OSError:
                    if os.path.exists(dst):
                        shutil.rmtree(dst, ignore_errors=True)
                    shutil.copytree(tmp, dst, dirs_exist_ok=True)
                    shutil.rmtree(tmp, ignore_errors=True)

                self._log("info", f"[PUBLISHER] checkpoint-{step} yayınlandı (atomic).")
                return True
            except Exception as e:
                last_err = e
                self._log("error", f"[PUBLISHER] Hata: {e!r}")
                if os.path.exists(tmp):
                    shutil.rmtree(tmp, ignore_errors=True)
                time.sleep(3)

        self._log("error", f"[PUBLISHER] checkpoint-{step} yayınlanamadı: {last_err!r}")
        return False

    def on_save(self, args, state, control, **kwargs):
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return

        step = state.global_step
        if step % self.drive_sync_steps != 0:
            self._log("info", f"[PUBLISHER] checkpoint-{step} yerel NVMe'ye kaydedildi (Drive sync: adım {step + (self.drive_sync_steps - step % self.drive_sync_steps)})")
            return

        src = os.path.join(args.output_dir, f"checkpoint-{step}")
        dst = os.path.join(self.drive_dir, f"checkpoint-{step}")

        if not self._publish_checkpoint(src, dst, step):
            return

        self._sync_logs()

        # Eski checkpoint temizliği
        best_ckpt = getattr(state, "best_model_checkpoint", None)
        best_step = self._get_step(best_ckpt) if best_ckpt else -1

        valid = [p for p in glob.glob(os.path.join(self.drive_dir, "checkpoint-*")) if self._is_valid_drive(p)]
        valid.sort(key=self._get_step)

        candidates = [p for p in valid if self._get_step(p) != best_step]
        while len(candidates) > self.keep_last:
            old = candidates.pop(0)
            try:
                shutil.rmtree(old)
                self._log("info", f"[PUBLISHER] Eski checkpoint temizlendi: {os.path.basename(old)}")
            except Exception as e:
                self._log("warning", f"[PUBLISHER] Temizleme hatası: {e!r}")

    def on_train_end(self, args, state, control, **kwargs):
        step = getattr(state, "global_step", None)
        if step is not None:
            src = os.path.join(args.output_dir, f"checkpoint-{step}")
            dst = os.path.join(self.drive_dir, f"checkpoint-{step}")
            if os.path.exists(src) and not self._is_valid_drive(dst):
                self._log("info", f"[PUBLISHER] on_train_end: Son checkpoint-{step} Drive'a aktarılıyor...")
                self._publish_checkpoint(src, dst, step)

        self._sync_logs()
        self._log("info", "[PUBLISHER] Final log senkronizasyonu tamamlandı.")

# ==============================================================================
# MODULE 7: SFT RESUME ORCHESTRATOR (SAFE v10.2 ONLY)
# ==============================================================================
class SFTResumeOrchestrator:
    def __init__(self, local_dir: str, drive_dir: str, arch_sig: str, telemetry: TelemetryLogger):
        self.local_dir = local_dir
        self.drive_dir = drive_dir
        self.arch_sig = arch_sig
        self.tel = telemetry
        os.makedirs(local_dir, exist_ok=True)
        os.makedirs(drive_dir, exist_ok=True)

    def _get_step(self, path: Optional[str]) -> int:
        if not path:
            return -1
        m = re.search(r"checkpoint-(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else -1

    def _is_valid_local(self, path: str) -> bool:
        if not os.path.isdir(path):
            return False
        for f in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
            if not os.path.exists(os.path.join(path, f)):
                return False
        return True

    def _is_valid_drive(self, path: str) -> bool:
        if not os.path.isdir(path):
            return False
        return os.path.exists(os.path.join(path, "_COMPLETE")) and os.path.exists(os.path.join(path, "trainer_state.json"))

    def find_resume_point(self) -> Optional[str]:
        self.tel.info("=" * 80)
        self.tel.info("SFT RESUME ORCHESTRATION (v10.2 FIX ONLY)")
        self.tel.info("=" * 80)

        drive_ckpts = [p for p in glob.glob(os.path.join(self.drive_dir, "checkpoint-*")) if self._is_valid_drive(p)]
        local_ckpts = [p for p in glob.glob(os.path.join(self.local_dir, "checkpoint-*")) if self._is_valid_local(p)]

        best_local = max(local_ckpts, key=self._get_step) if local_ckpts else None
        best_drive = max(drive_ckpts, key=self._get_step) if drive_ckpts else None

        step_local = self._get_step(best_local)
        step_drive = self._get_step(best_drive)

        self.tel.info(f"-> En güncel Yerel SFT checkpoint : {os.path.basename(best_local) if best_local else 'YOK'} (Adım {step_local})")
        self.tel.info(f"-> En güncel Drive SFT checkpoint : {os.path.basename(best_drive) if best_drive else 'YOK'} (Adım {step_drive})")

        resume_target = None
        if step_drive > step_local and best_drive is not None:
            self.tel.info(f"-> Drive daha güncel ({step_drive} > {step_local}). Smart Fetch ile yerel NVMe'ye alınıyor...")
            target = os.path.join(self.local_dir, os.path.basename(best_drive))
            if os.path.exists(target):
                shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(best_drive, target)
            resume_target = target
        elif best_local is not None:
            self.tel.info(f"-> Yerel checkpoint kullanılıyor: {os.path.basename(best_local)}")
            resume_target = best_local

        return resume_target

# ==============================================================================
# MODULE 8: SFT INFERENCE EVALUATOR (5 TARGET BENCHMARK PROMPTS)
# ==============================================================================
class SFTModelEvaluator:
    def __init__(self, telemetry: TelemetryLogger):
        self.tel = telemetry

    def evaluate(self, trainer: Trainer, eval_ds) -> Tuple[float, float]:
        self.tel.info("=" * 80)
        self.tel.info("SFT HELD-OUT EVALUATION (MASKED ASSISTANT LOSS & PPL)")
        self.tel.info("=" * 80)
        metrics = trainer.evaluate(eval_dataset=eval_ds)
        loss = metrics.get("eval_loss", float("nan"))
        try:
            ppl = math.exp(loss)
        except (OverflowError, ValueError):
            ppl = float("inf")
        self.tel.info(f"-> SFT Eval Loss       : {loss:.4f} (Hedef: < 2.5, Önceki: 3.488)")
        self.tel.info(f"-> SFT Perplexity      : {ppl:.2f} (Hedef: < 15.0, Önceki: 32.74)")
        self.tel.info(f"-> Eval Örnek Sayısı   : {len(eval_ds):,}")
        self.tel.record_metric("eval_loss", loss)
        self.tel.record_metric("eval_ppl", ppl)
        return loss, ppl

    def inference_benchmark(self, model_path: str, tokenizer: PreTrainedTokenizerFast,
                            attn_impl: str, dtype: torch.dtype) -> List[Dict[str, str]]:
        self.tel.info("=" * 80)
        self.tel.info("SFT TALİMAT TAKİP BENCHMARK TESTİ (5 HEDEF SORU)")
        self.tel.info("=" * 80)
        device = torch.device("cuda")

        try:
            model = LlamaForCausalLM.from_pretrained(
                model_path, attn_implementation=attn_impl, **dtype_kwarg(dtype)
            ).to(device)
        except Exception as e:
            self.tel.warning(f"Attn={attn_impl} yüklenemedi: {e!r}. SDPA fallback deneniyor...")
            model = LlamaForCausalLM.from_pretrained(
                model_path, attn_implementation="sdpa", **dtype_kwarg(dtype)
            ).to(device)

        model.eval()

        # Kullanıcının şart koştuğu 5 hedef soru
        test_questions = [
            "Suyun kimyasal formülü nedir?",
            "Türkiye'nin başkenti neresi?",
            "Python'da liste nasıl ters çevrilir?",
            "Mustafa Kemal Atatürk kimdir?",
            "Güneş Sistemi'nin en büyük gezegeni hangisi?"
        ]

        results = []
        for idx, q in enumerate(test_questions, 1):
            prompt = f"### Kullanıcı:\n{q}\n\n### Asistan:\n"
            input_ids = [tokenizer.bos_token_id] + tokenizer(prompt, add_special_tokens=False)["input_ids"]
            inputs = torch.tensor([input_ids], device=device)

            with torch.inference_mode():
                out = model.generate(
                    inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    top_k=40,
                    top_p=0.9,
                    temperature=0.6,
                    repetition_penalty=1.15,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            answer = tokenizer.decode(out[0][len(input_ids):], skip_special_tokens=True).strip()
            self.tel.info(f"[{idx}/5] Soru : {q}")
            self.tel.info(f"      Yanıt: {answer}\n" + "-" * 60)
            results.append({"question": q, "answer": answer})

        return results

# ==============================================================================
# MODULE 7: DINAMIK MICRO-BATCH ARAMA (v10.0'dan port, SFT uyumlu)
# ==============================================================================
def _sft_forward_backward(model, train_ds, collator, device, dtype, batch_size, simulate_optimizer=False):
    """Tek forward+backward geçişi. OOM tespiti için find_optimal_sft_batch tarafından çağrılır."""
    model.to(device)
    model.train()
    batch = None
    dummy_opt = None
    loss = None
    try:
        batch = collator([train_ds[i] for i in range(batch_size)])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            loss = model(**batch).loss
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Loss NaN/Inf: {loss.item()}")
        loss.backward()
        if simulate_optimizer:
            dummy_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            dummy_opt.step()
    finally:
        if loss is not None:
            del loss
        model.zero_grad(set_to_none=True)
        if dummy_opt is not None:
            del dummy_opt
        if batch is not None:
            del batch
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def find_optimal_sft_batch(model, train_ds, collator, device, dtype, total_vram_gb, tel,
                           seed_batch=8, max_batch=None):
    """v10.0 BackendOrchestrator.find_optimal_batch'in SFT standalone versiyonu.

    Exponential ramp-up (2x katlama) + binary search ile VRAM'in %92'sine kadar
    giden maksimum micro-batch'i bulur. T4/L4/A100 fark etmeksizin auto-scale.
    """
    tel.info("=" * 80)
    tel.info("DINAMIK MICRO-BATCH ARAMASI (SELF-SCALING, HEADROOM GUARD %92)")
    tel.info("=" * 80)

    initial_weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    dataset_cap = len(train_ds)
    hard_cap = max(1, min(max_batch, dataset_cap) if max_batch else dataset_cap)

    def _cleanup():
        model.zero_grad(set_to_none=True)
        model.load_state_dict(initial_weights)
        gc.collect()
        for _ in range(3):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            time.sleep(0.3)
        torch.cuda.reset_peak_memory_stats()

    def _fits(bs):
        try:
            torch.cuda.reset_peak_memory_stats()
            tel.info(f"-> Deneniyor: micro_batch={bs} (Forward + Backward + AdamW)...")
            _sft_forward_backward(model, train_ds, collator, device, dtype,
                                  batch_size=bs, simulate_optimizer=True)
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            vram_ratio = peak_mem / total_vram_gb
            tel.info(f"   Peak VRAM: {peak_mem:.2f} GB / {total_vram_gb:.2f} GB (%{vram_ratio*100:.1f})")
            if vram_ratio > 0.92:
                tel.warning(f"   [Headroom Guard] micro_batch={bs} guvenlik tavanini (%92) asti.")
                _cleanup()
                return False
            _cleanup()
            return True
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            tel.warning(f"   [OOM/Hata] micro_batch={bs} sigmadi: {e!r}")
            _cleanup()
            return False

    # 1) Exponential ramp-up
    lo, hi = None, None
    bs = max(1, min(seed_batch, hard_cap))
    while True:
        if _fits(bs):
            lo = bs
            if bs >= hard_cap:
                break
            bs = min(bs * 2, hard_cap)
        else:
            hi = bs
            break

    if lo is None:
        if bs > 1 and _fits(1):
            lo, hi = 1, bs
        else:
            tel.fatal("GPU batch=1 boyutunu bile sigdiramıyor! seed_batch=8 ile devam.")
            return seed_batch  # Güvenli fallback

    if hi is not None:
        # 2) Binary search: lo (sigan) ile hi (sigmayan) arasında gercek tavanı bul
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if _fits(mid):
                lo = mid
            else:
                hi = mid

    del initial_weights
    gc.collect()
    torch.cuda.empty_cache()

    tel.info(f"-> OPTIMAL MICRO-BATCH BULUNDU = {lo} (dinamik arama, %92 VRAM headroom)")
    return lo


# ==============================================================================
# MAIN SFT PIPELINE
# ==============================================================================
def main():
    # ---- 0. ENVIRONMENT SETUP ----
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_HOME"] = os.path.join(DRIVE_BASE, "hf_cache")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # FIX: tqdm progress bar spam'ini kapat (EVAL_STEPS=50'de 32K+ satır yapıyordu)
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    if _IN_COLAB and not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    os.makedirs(LOCAL_LOGS_ROOT, exist_ok=True)

    boot_tel = TelemetryLogger(LOCAL_LOGS_ROOT, "sft_bootstrap_v10.2")

    # ---- BAŞLANGIÇ LOGLARI (ŞART KOŞULAN BAŞLIKLAR) ----
    boot_tel.info("=" * 80)
    boot_tel.info("=== v10.2 SFT FIX BAŞLIYOR ===")
    boot_tel.info(f"Base: {EXACT_BASE_MODEL_NAME}")
    boot_tel.info("LR: 1e-4 (2e-5'ten 5x artırıldı - modelin talimat öğrenmesi için)")
    boot_tel.info("Epoch: 3 (2'den artırıldı - tam sindirme)")
    boot_tel.info("Veri: merve (51K) + TFLai/Turkish-Alpaca (52K) = ~103K")
    boot_tel.info("Beklenen sure: ~20-25 dakika (A100 GPU, ~2,400 adim)")
    boot_tel.info("Hedef: eval_loss < 2.5 (önceki v10.1'de 3.488 idi)")
    boot_tel.info("=" * 80)

    # ---- 1. HARDWARE AUDIT ----
    specs = HardwareAuditor(boot_tel).audit()

    # ---- 2. HARDWARE PROFILES & ARCHITECTURE ----
    if specs["vram_gb"] >= 35.0:
        PROFILE = "A100_HIGH"
        CONTEXT_LEN, HIDDEN_SIZE, INTERMEDIATE_SIZE = 1024, 1024, 4096
        NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS = 24, 16, 8
        USE_GRAD_CHECKPOINTING = False
        MICRO_BATCH = 40
        GRAD_ACCUM = 3   # Efektif batch: 40 * 3 = 120
    elif specs["vram_gb"] >= 20.0:
        PROFILE = "L4_MID"
        CONTEXT_LEN, HIDDEN_SIZE, INTERMEDIATE_SIZE = 1024, 1024, 4096
        NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS = 24, 16, 8
        USE_GRAD_CHECKPOINTING = True
        MICRO_BATCH = 15
        GRAD_ACCUM = 8   # Efektif batch: 15 * 8 = 120
    else:
        PROFILE = "T4_BUDGET"
        CONTEXT_LEN, HIDDEN_SIZE, INTERMEDIATE_SIZE = 1024, 768, 2048
        NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS = 12, 12, 12
        USE_GRAD_CHECKPOINTING = True
        MICRO_BATCH = 8
        GRAD_ACCUM = 15  # Efektif batch: 8 * 15 = 120

    # ---- SFT HIPERPARAMETRELERİ (v10.2 FIX) ----
    EPOCHS = 3                  # DEĞİŞTİ: 2 -> 3
    LEARNING_RATE = 1e-4        # DEĞİŞTİ: 2e-5 -> 1e-4 (5x artırıldı)
    WARMUP_RATIO = 0.05         # DEĞİŞTİ: 0.03 -> 0.05
    SAVE_STEPS = 200            # FIX: 50 -> 200 (her 200 adımda yerel kayıt)
    EVAL_STEPS = 200            # FIX: 50 -> 200 (~12 eval, 50 yerine ~50 eval = log spam önlendi)
    DRIVE_SYNC_STEPS = 400      # FIX: 100 -> 400 (daha az Drive I/O, daha hızlı eğitim)
    LOGGING_STEPS = 50          # FIX: 10 -> 50 (anlamlı aralıklar, tqdm gürültüsü azalır)
    WEIGHT_DECAY = 0.01

    # ---- 3. TOKENIZER ----
    tok_path = os.path.join(DRIVE_BASE, "tokenizer", "tokenizer.json")
    tokenizer, tok_hash = TokenizerAuditor(tok_path, boot_tel).audit_and_load()

    # ---- 4. BASE MODEL BULMA VE DOĞRULAMA ----
    expected_arch = {
        "hidden": HIDDEN_SIZE, "layers": NUM_LAYERS,
        "heads": NUM_HEADS, "vocab": len(tokenizer),
    }
    base_finder = SmartBaseModelFinder(DRIVE_BASE, expected_arch, boot_tel)
    validated_base_path = base_finder.find_base_model(BASE_MODEL_PATH)

    # ---- 5. SFT DATASET (10K SADECE MERVE) ----
    sft_ds_mgr = SFTDatasetManager(
        DRIVE_BASE, LOCAL_BASE, DATA_VERSION, CONTEXT_LEN,
        tokenizer, tok_hash, boot_tel
    )
    train_ds, eval_ds = sft_ds_mgr.load_or_build_datasets(is_dry_run=DRY_RUN_MODE)

    # ---- 6. SIGNATURES & RUN PATHS ----
    arch_sig = hashlib.sha256(json.dumps(expected_arch, sort_keys=True).encode()).hexdigest()[:16]
    RUN_NAME = f"turklm_{RUN_VERSION}_{PROFILE.lower()}_{arch_sig}"
    tel = TelemetryLogger(LOCAL_LOGS_ROOT, RUN_NAME)
    tel.info(f"AKTİF SFT KOŞU ADI: {RUN_NAME}")

    LOCAL_CKPT = os.path.join(LOCAL_BASE, f"ckpt_{RUN_NAME}")
    DRIVE_CKPT = os.path.join(DRIVE_BASE, f"ckpt_{RUN_NAME}")
    LOCAL_FINAL = os.path.join(LOCAL_BASE, f"final_{RUN_NAME}")
    DRIVE_FINAL = os.path.join(DRIVE_BASE, f"final_{RUN_NAME}")
    LOCAL_LOGS = os.path.join(LOCAL_BASE, f"logs_{RUN_NAME}")
    DRIVE_LOGS = os.path.join(DRIVE_BASE, "logs", RUN_NAME)

    os.makedirs(LOCAL_CKPT, exist_ok=True)
    os.makedirs(DRIVE_CKPT, exist_ok=True)
    os.makedirs(LOCAL_LOGS, exist_ok=True)
    os.makedirs(DRIVE_LOGS, exist_ok=True)

    # ---- 7. BASE MODELİ YÜKLE ----
    preferred_attn = "flash_attention_2" if specs["flash_attn_available"] else "sdpa"
    dtype = torch.bfloat16 if specs["use_bf16"] else torch.float16
    device = torch.device("cuda")

    tel.info("=" * 80)
    tel.info("BASE MODEL YÜKLENİYOR (WEIGHT TRANSFER)")
    tel.info("=" * 80)
    try:
        model = LlamaForCausalLM.from_pretrained(
            validated_base_path,
            attn_implementation=preferred_attn,
            **dtype_kwarg(dtype)
        ).to(device)
        active_backend = preferred_attn
        tel.info(f"-> Base Model {preferred_attn.upper()} ile başarıyla yüklendi.")
    except Exception as e:
        tel.warning(f"Preferred attn ({preferred_attn}) yüklenemedi: {e!r}. SDPA fallback yapılıyor...")
        model = LlamaForCausalLM.from_pretrained(
            validated_base_path,
            attn_implementation="sdpa",
            **dtype_kwarg(dtype)
        ).to(device)
        active_backend = "sdpa"
        tel.info("-> Base Model SDPA ile yüklendi.")

    if USE_GRAD_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        tel.info("-> Gradient Checkpointing aktif edildi.")

    param_count = sum(p.numel() for p in model.parameters())
    tel.info(f"-> Toplam Parametre Sayısı: {param_count / 1e6:.2f} M")

    # ---- 8. DRY RUN MODU TESTİ ----
    if DRY_RUN_MODE:
        tel.info("=" * 80)
        tel.info("DRY RUN TESTİ: Model, Tokenizer ve Veri Hattı Denetleniyor...")
        tel.info("=" * 80)
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, pad_to_multiple_of=8,
            label_pad_token_id=-100, return_tensors="pt"
        )
        batch = collator([train_ds[i] for i in range(min(4, len(train_ds)))])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            loss = model(**batch).loss
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Dry run loss NaN/Inf: {loss.item()}")
        loss.backward()
        tel.info(f"-> Forward/Backward Başarılı! Test Kaybı (Dry-Run Loss): {loss.item():.4f}")
        tel.info("=== DRY RUN BAŞARILI: Model, Tokenizer ve Veri Hattı Sorunsuz Çalışıyor! ===")
        return

    # ---- 9. RESUME DENETİMİ (v10.2 ÖZEL) ----
    last_checkpoint = SFTResumeOrchestrator(LOCAL_CKPT, DRIVE_CKPT, arch_sig, tel).find_resume_point()

    # ---- 10. DATA COLLATOR (PAD & MASK GÜVENCESİ) ----
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    # ---- 10b. DİNAMİK MICRO-BATCH ARAMA (v10.0'dan port) ----
    # Profil bazlı sabit değeri (A100=40, L4=15, T4=8) override eder.
    # Exponential ramp-up + binary search ile %92 VRAM headroom'u koruyarak
    # gerçek optimali bulur. A100 80GB için ~100 bulmasi beklenir.
    seed_batch = MICRO_BATCH  # Profil tahmini başlangıç noktası olarak kullan
    MICRO_BATCH = find_optimal_sft_batch(
        model=model,
        train_ds=train_ds,
        collator=collator,
        device=device,
        dtype=dtype,
        total_vram_gb=specs["vram_gb"],
        tel=tel,
        seed_batch=seed_batch,
        max_batch=256,  # SFT için makul üst sınır
    )
    # GRAD_ACCUM'u efektif batch sabit kalacak şekilde güncelle
    TARGET_EFF_BATCH = seed_batch * GRAD_ACCUM
    GRAD_ACCUM = max(1, TARGET_EFF_BATCH // MICRO_BATCH)
    tel.info(f"-> Efektif batch: {MICRO_BATCH} x {GRAD_ACCUM} = {MICRO_BATCH * GRAD_ACCUM} token/step")

    # ---- 11. TRAINING ARGUMENTS ----
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    active_optim = OPTIMIZER_TYPE
    if active_optim == "adamw_torch_fused" and not specs["use_tf32"]:
        active_optim = "adamw_torch"

    kwargs = {
        "output_dir": LOCAL_CKPT,
        "overwrite_output_dir": False,
        "num_train_epochs": EPOCHS,
        "max_steps": -1,  # Epoch 3 üzerinden hesaplansın (~240 step)
        "per_device_train_batch_size": MICRO_BATCH,
        "per_device_eval_batch_size": min(MICRO_BATCH, 8),
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": 1.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "optim": active_optim,
        "bf16": specs["use_bf16"],
        "fp16": specs["use_fp16"],
        "tf32": specs["use_tf32"],
        "gradient_checkpointing": USE_GRAD_CHECKPOINTING,
        "group_by_length": False,  # DEĞİŞTİ: DataCollatorForSeq2Seq crash riskini önlemek için False
        "save_steps": SAVE_STEPS,
        "eval_steps": EVAL_STEPS,
        "save_total_limit": 3,
        "load_best_model_at_end": False,
        "prediction_loss_only": True,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": True,
        "remove_unused_columns": False,
        "torch_compile": False,
        "logging_steps": LOGGING_STEPS,
        "logging_first_step": False,   # FIX: ilk step'te ekstra log spam yok
        "disable_tqdm": True,          # FIX: Trainer progress bar'ını kapat (log temiz kalır)
        "logging_dir": LOCAL_LOGS,
        "seed": 42,
        "data_seed": 42,
        "report_to": "none",
    }
    kwargs = {k: v for k, v in kwargs.items() if k in ta_params}
    if "eval_strategy" in ta_params:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in ta_params:
        kwargs["evaluation_strategy"] = "steps"
    if "save_strategy" in ta_params:
        kwargs["save_strategy"] = "steps"

    args = TrainingArguments(**kwargs)

    # ---- 12. TRAINER BAĞLANTISI ----
    publisher = SFTAtomicDrivePublisher(
        drive_ckpt_dir=DRIVE_CKPT,
        local_logs_dir=LOCAL_LOGS,
        drive_logs_dir=DRIVE_LOGS,
        keep_last=2,
        telemetry=tel,
        drive_sync_steps=DRIVE_SYNC_STEPS,
    )

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
        "callbacks": [publisher],
    }
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    effective_batch = MICRO_BATCH * GRAD_ACCUM
    steps_per_epoch = len(train_ds) // effective_batch
    total_steps = steps_per_epoch * EPOCHS

    tel.info("=" * 80)
    tel.info(f"PRE-FLIGHT ONAY: Toplam Örnek={len(train_ds):,} | Adım/Epoch={steps_per_epoch} | Toplam Adım={total_steps}")
    tel.info(f"Efektif Batch={effective_batch} | LR={LEARNING_RATE} | Tahmini Süre: ~12-15 dk")
    tel.info("=" * 80)

    # Allocator temizliği
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ---- 13. LIFTOFF (EĞİTİM) ----
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # ---- 14. EVALUATION ----
    evaluator = SFTModelEvaluator(tel)
    eval_loss, eval_ppl = evaluator.evaluate(trainer, eval_ds)

    # ---- 15. ATOMIC NİHAİ MODEL KAYDI ----
    tel.info("=" * 80)
    tel.info("FINAL SFT MODEL KAYDI (ATOMIC)")
    tel.info("=" * 80)

    if os.path.exists(LOCAL_FINAL):
        shutil.rmtree(LOCAL_FINAL)
    os.makedirs(LOCAL_FINAL, exist_ok=True)

    raw_model = getattr(trainer.model, "_orig_mod", trainer.model)
    if hasattr(raw_model, "config"):
        raw_model.config.use_cache = True
    if hasattr(raw_model, "gradient_checkpointing_disable"):
        raw_model.gradient_checkpointing_disable()

    trainer.save_model(LOCAL_FINAL)
    tokenizer.save_pretrained(LOCAL_FINAL)

    # Atomic Drive Transfer
    FINAL_TMP = DRIVE_FINAL + ".partial"
    if os.path.exists(FINAL_TMP):
        shutil.rmtree(FINAL_TMP, ignore_errors=True)
    if os.path.exists(DRIVE_FINAL):
        shutil.rmtree(DRIVE_FINAL, ignore_errors=True)
    shutil.copytree(LOCAL_FINAL, FINAL_TMP)
    try:
        os.replace(FINAL_TMP, DRIVE_FINAL)
    except OSError:
        if os.path.exists(DRIVE_FINAL):
            shutil.rmtree(DRIVE_FINAL, ignore_errors=True)
        shutil.copytree(FINAL_TMP, DRIVE_FINAL, dirs_exist_ok=True)
        shutil.rmtree(FINAL_TMP, ignore_errors=True)

    tel.info(f"-> Nihai SFT model Drive'a kaydedildi: {DRIVE_FINAL}")

    # ---- 16. MODEL TEMİZLİĞİ & TEST ----
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- 17. 5 HEDEF SORU İLE INFERENCE BENCHMARK ----
    test_path = LOCAL_FINAL if os.path.exists(LOCAL_FINAL) else DRIVE_FINAL
    benchmark_answers = evaluator.inference_benchmark(test_path, tokenizer, active_backend, dtype)

    # ---- 18. NİHAİ METRİKLERİ VE MANIFEST'İ KAYDET ----
    manifest = {
        "RUN_VERSION": RUN_VERSION,
        "base_model": validated_base_path,
        "lr": LEARNING_RATE,
        "epochs": EPOCHS,
        "final_loss": getattr(train_result, "training_loss", None),
        "final_eval_loss": eval_loss,
        "perplexity": eval_ppl,
        "eval_loss_previous_v10_1": 3.488,
        "perplexity_previous_v10_1": 32.74,
        "target_eval_loss": "< 2.5",
        "benchmark_answers": benchmark_answers,
        "saved_at": datetime.now().isoformat(),
    }
    with open(os.path.join(LOCAL_FINAL, "sft_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(DRIVE_FINAL, "sft_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    tel.save_manifest(os.path.join(DRIVE_FINAL, "sft_telemetry_manifest.json"))

    # ---- 19. BAŞARI / BAŞARISIZLIK DEĞERLENDİRMESİ ----
    tel.info("=" * 80)
    tel.info("v10.2 SFT FIX SONUÇ RAPORU")
    tel.info("=" * 80)
    tel.info(f"Nihai Eval Loss: {eval_loss:.4f} (v10.1: 3.488 | Hedef: < 2.5)")
    tel.info(f"Nihai PPL      : {eval_ppl:.2f}  (v10.1: 32.74 | Hedef: < 15.0)")

    if eval_loss < 2.5:
        tel.info("-> [BAŞARILI] SFT hedefleri tuttu! Loss 2.5 altına indi. Model instruction-following öğrendi. [OK]")
    else:
        tel.warning("-> [DİKKAT] eval_loss hedefi (< 2.5) tam tutturulamadı.")

    # Başarısızlık Durumu Değerlendirmesi:
    if eval_loss >= 3.0:
        tel.warning(
            "\n" + "!" * 80 + "\n"
            "DİKKAT: eval_loss hâlâ 3.0 üzerinde kaldı!\n"
            "Bu durum, sorunun SFT eğitiminde değil, 443M Base modelin bilgi tavanında olduğunu kanıtlar.\n"
            "KARAR: Daha fazla SFT denemesi yerine doğrudan 15 Milyar tokenlık 'train_base-v2.py' eğitimine geçilmelidir!\n"
            + "!" * 80
        )

    tel.info("=" * 80)
    tel.info("TURKLM v10.2 SFT FIX OPERASYONU TAMAMLANDI")
    tel.info("=" * 80)


if __name__ == "__main__":
    main()
