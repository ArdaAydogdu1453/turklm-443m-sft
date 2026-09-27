# ==============================================================================
# TURKLM v10.0_turbo — PRODUCTION MASTER (LIGER + COMPILE + SPEED OPTIMIZED)
# ==============================================================================
# ==============================================================================
# NOT: Bu kod Türkçe LLM eğitim pipeline'ı.
# Çalıştırmak için önce kendi Drive'ında:
#   1. /content/drive/MyDrive/Turkce_Tiny_LM/tokenizer/tokenizer.json
#   2. /content/drive/MyDrive/Turkce_Tiny_LM/dataset_*/ klasörleri olmalı.
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
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import sys
import subprocess
import torch
try:
    import torch._dynamo
except ImportError:
    pass

# ==============================================================================
# FLASH ATTENTION 2 OTOMATİK KURULUM (Liger'den önce, çakışmaz)
# ==============================================================================
def _init_flash_attention() -> bool:
    """flash-attn kurulu mu diye kontrol eder. Kurmaya ÇALIŞMAZ —
    kurulum ayrı hücrede manuel yapılır. Böylece 3 dk timeout beklemek yok."""
    try:
        import flash_attn  # noqa: F401
        print("[FLASH-ATTN] Kurulu, kullanılacak.")
        return True
    except ImportError:
        print("[FLASH-ATTN] Kurulu değil → SDPA kullanılacak.")
        return False
    except Exception as e:
        print(f"[FLASH-ATTN] Kontrol hatası: {e!r}")
        return False

_init_flash_attention()

# ==============================================================================
# LIGER KERNEL OTOMATİK KURULUM VE LLaMA MONKEY-PATCH
# transformers modelleri yüklenmeden ÖNCE çağrılmalıdır.
# ==============================================================================
_LIGER_ACTIVE = False
def _init_liger_kernel():
    global _LIGER_ACTIVE
    disable_loss = os.environ.get("LIGER_KERNEL_DISABLE_LOSS", "0") == "1"
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_llama
        if disable_loss:
            apply_liger_kernel_to_llama(cross_entropy=False, fused_linear_cross_entropy=False)
        else:
            apply_liger_kernel_to_llama()
        _LIGER_ACTIVE = True
        print("[LIGER] Aktif — hız optimizasyonları devrede.")
    except ImportError:
        print("[LIGER] Kurulu değil, SDPA+standart PyTorch ile devam.")
        _LIGER_ACTIVE = False
    except Exception as e:
        print(f"[LIGER] Aktivasyon hatası: {e!r}")
        _LIGER_ACTIVE = False

_init_liger_kernel()

from google.colab import drive
from transformers import (
    TrainerCallback,
    LlamaConfig,
    LlamaForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerFast
)

# ==============================================================================
# MODULE 0: GLOBAL CONSTANTS & HELPERS
# ==============================================================================
DRIVE_BASE = "/content/drive/MyDrive/Turkce_Tiny_LM"
LOCAL_BASE = "/content"
LOCAL_LOGS_ROOT = "/content/logs_telemetry"
DATA_VERSION = "dedup_packed_v4_6"
RUN_VERSION = "v10.0_turbo"

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
        self.tel.info("HARDWARE & ENVIRONMENT AUDIT")
        self.tel.info("=" * 80)

        if not torch.cuda.is_available():
            self.tel.fatal("CUDA GPU bulunamadı! Runtime > Change runtime type > GPU seçin.")

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

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            occupied_gb = (total_bytes - free_bytes) / (1024 ** 3)
            if occupied_gb > 3.0:
                self.tel.warning(
                    f"DİKKAT: GPU VRAM'inde önceden kalan {occupied_gb:.1f} GB dolu alan tespit edildi!\n"
                    f"   (Önceki çalışmadan kalan CUDA Graph / önbellek kilitlenmiş olabilir).\n"
                    f"   Lütfen Colab menüsünden: 'Runtime -> Restart Session' (Oturumu Yeniden Başlat) yapın!"
                )
        except Exception:
            pass

        self.tel.info("HARDWARE AUDIT PASSED.")
        return specs

# ==============================================================================
# MODULE 3: IMMUTABLE TOKENIZER CONTRACT
# ==============================================================================
class TokenizerAuditor:
    def __init__(self, tokenizer_path: str, telemetry: TelemetryLogger):
        self.path = tokenizer_path
        self.tel = telemetry
        self.required_tokens = {
            "bos_token": "<s>", "eos_token": "</s>",
            "unk_token": "<unk>", "pad_token": "<pad>",
        }

    def audit_and_load(self) -> Tuple[PreTrainedTokenizerFast, str]:
        self.tel.info("=" * 80)
        self.tel.info("TOKENIZER IMMUTABILITY AUDIT")
        self.tel.info("=" * 80)

        if not os.path.exists(self.path):
            self.tel.fatal(f"Tokenizer artifact missing: {self.path}")

        with open(self.path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:12]
        self.tel.info(f"-> Tokenizer SHA-256: {file_hash}")
        self.tel.record_metric("tokenizer_hash", file_hash)

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self.path,
            bos_token="<s>", eos_token="</s>",
            unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
        )

        for name, expected in self.required_tokens.items():
            token_id = getattr(tokenizer, f"{name}_id", None)
            if token_id is None:
                self.tel.fatal(f"IMMUTABLE CONTRACT VIOLATED: {name} ({expected!r}) eksik!")
            self.tel.info(f"-> Doğrulandı: {name:<12} = ID {token_id}")

        vocab_size = len(tokenizer)
        self.tel.info(f"-> Toplam Vocab size: {vocab_size}")
        self.tel.record_metric("vocab_size", vocab_size)
        self.tel.info("TOKENIZER AUDIT PASSED.")
        return tokenizer, file_hash

# ==============================================================================
# MODULE 4: DATASET MANAGER
# ==============================================================================
class DatasetManager:
    def __init__(self, drive_base: str, local_base: str, data_version: str,
                 context_len: int, tokenizer_hash: str, telemetry: TelemetryLogger):
        self.tel = telemetry
        suffix = f"dataset_{data_version}_ctx{context_len}_{tokenizer_hash}"
        self.drive_cache = os.path.join(drive_base, suffix)
        self.local_cache = os.path.join(local_base, f"local_{suffix}")

    def load_datasets(self):
        self.tel.info("=" * 80)
        self.tel.info("DATASET CACHE INTEGRITY CHECK")
        self.tel.info("=" * 80)
        self.tel.info(f"-> Drive cache: {self.drive_cache}")
        self.tel.info(f"-> Local cache: {self.local_cache}")

        from datasets import Dataset, DatasetDict
        drive_marker = os.path.join(self.drive_cache, "_COMPLETE")
        local_marker = os.path.join(self.local_cache, "_COMPLETE")

        if not os.path.exists(drive_marker):
            self.tel.fatal(f"Doğrulanmış veri cache'i bulunamadı: {drive_marker}")

        if not os.path.exists(local_marker):
            self.tel.info("-> Local NVMe cache yok. Drive'dan kopyalanıyor...")
            if os.path.exists(self.local_cache):
                shutil.rmtree(self.local_cache, ignore_errors=True)
            shutil.copytree(self.drive_cache, self.local_cache)
            self.tel.info("-> NVMe kopyalama tamamlandı.")
        else:
            self.tel.info("-> Doğrulanmış local NVMe cache bulundu.")

        train_path = os.path.join(self.local_cache, "train")
        eval_path = os.path.join(self.local_cache, "eval")

        if os.path.exists(train_path) and os.path.exists(eval_path):
            train_ds = Dataset.load_from_disk(train_path)
            eval_ds = Dataset.load_from_disk(eval_path)
        else:
            dd = DatasetDict.load_from_disk(self.local_cache)
            train_ds, eval_ds = dd["train"], dd["eval"]

        self.tel.info(f"-> Train blok sayısı : {len(train_ds):,}")
        self.tel.info(f"-> Eval blok sayısı  : {len(eval_ds):,}")
        self.tel.record_metric("train_blocks", len(train_ds))
        self.tel.record_metric("eval_blocks", len(eval_ds))
        self.tel.info("DATASET INTEGRITY PASSED.")
        return train_ds, eval_ds

# ==============================================================================
# MODULE 5: BACKEND ORCHESTRATOR & DYNAMIC BATCH SEARCH (LEAK-PROOF)
# ==============================================================================
class BackendOrchestrator:
    def __init__(self, hardware_specs: Dict[str, Any], telemetry: TelemetryLogger):
        self.specs = hardware_specs
        self.tel = telemetry
        self.device = torch.device("cuda")
        self._FLASH_KEYWORDS = (
            "flash", "flashattention", "no available kernel",
            "unsupported by flashattention", "fa2",
        )

    def _is_flash_error(self, err: Exception) -> bool:
        msg = repr(err).lower()
        return any(k in msg for k in self._FLASH_KEYWORDS)

    def _build_model(self, config: LlamaConfig, backend: str) -> LlamaForCausalLM:
        config._attn_implementation = backend
        try:
            return LlamaForCausalLM(config, attn_implementation=backend)
        except TypeError:
            return LlamaForCausalLM(config)

    def _forward_backward(self, model: torch.nn.Module, train_ds, collator,
                          batch_size: int, simulate_optimizer: bool = False):
        model.to(self.device)
        model.train()
        batch = None
        dummy_opt = None
        loss = None
        try:
            batch = collator([train_ds[i] for i in range(batch_size)])
            batch = {k: v.to(self.device) for k, v in batch.items()}
            dtype = torch.bfloat16 if self.specs["use_bf16"] else torch.float16

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
                loss = model(**batch).loss

            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Loss NaN/Inf tespit edildi: {loss.item()}")

            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise RuntimeError(f"NaN/Inf gradient: {name}")

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

    def _select_backend(self, config: LlamaConfig, train_ds, collator) -> Tuple[torch.nn.Module, str]:
        preferred = "flash_attention_2" if self.specs["flash_attn_available"] else "sdpa"
        self.tel.info(f"-> Tercih edilen dikkat mekanizması: {preferred.upper()}")

        model = None
        first_error: Optional[Exception] = None

        try:
            model = self._build_model(config, preferred)
            self._forward_backward(model, train_ds, collator, batch_size=2, simulate_optimizer=False)
            self.tel.info(f"-> Backend testi BAŞARILI: {preferred.upper()}")
            return model, preferred
        except Exception as e:
            first_error = e

        if preferred == "flash_attention_2" and self._is_flash_error(first_error):
            self.tel.warning(f"Flash Attention kernel hatası: {first_error!r}. SDPA fallback devreye giriyor...")
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            try:
                model = self._build_model(config, "sdpa")
                self._forward_backward(model, train_ds, collator, batch_size=2, simulate_optimizer=False)
                self.tel.info("-> SDPA fallback testi BAŞARILI.")
                return model, "sdpa"
            except Exception as sdpa_err:
                self.tel.fatal(f"SDPA fallback başarısız: {sdpa_err!r}")

        self.tel.fatal(f"Kritik Backend Başlatma Hatası: {first_error!r}")
        raise RuntimeError(first_error)

    def find_optimal_batch(self, model: torch.nn.Module, train_ds, collator,
                           seed_batch: int = 2, max_batch: Optional[int] = None) -> int:
        """Sabit bir hedef tahmin etmek yerine, önce yukarı doğru katlayarak (2x)
        donanımın taşıyabildiği yeri keşfeder; OOM'a çarpınca son sığan (lo) ile
        ilk sığmayan (hi) arasında ikili arama yaparak tam sınırı bulur.
        T4, L4, A100 fark etmeksizin — hangi GPU'ya denk gelirse gelsin — aynı
        kod, o GPU'nun kaldırabildiği maksimuma kendini genişletir/daraltır."""
        self.tel.info("=" * 80)
        self.tel.info("DİNAMİK MICRO-BATCH ARAMASI (SELF-SCALING, HEADROOM GUARD %92)")
        self.tel.info("=" * 80)

        initial_weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        total_vram = self.specs["vram_gb"]
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

        def _fits(bs: int) -> bool:
            try:
                torch.cuda.reset_peak_memory_stats()
                self.tel.info(f"-> Deneniyor: micro_batch={bs} (Forward + Backward + AdamW)...")
                self._forward_backward(model, train_ds, collator, batch_size=bs, simulate_optimizer=True)
                peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                vram_ratio = peak_mem / total_vram
                self.tel.info(f"   Peak VRAM: {peak_mem:.2f} GB / {total_vram:.2f} GB (%{vram_ratio*100:.1f})")
                if vram_ratio > 0.92:
                    self.tel.warning(f"   [Headroom Guard] micro_batch={bs} güvenlik tavanını (%92) aştı.")
                    _cleanup()
                    return False
                _cleanup()
                return True
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                self.tel.warning(f"   [OOM/Hata] micro_batch={bs} sığmadı: {e!r}")
                _cleanup()
                return False

        # 1) EXPONENTIAL RAMP-UP: donanımın izin verdiği yere kadar 2x katla
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
            # seed_batch>1 ile başlanıp ilk denemede OOM olduysa, batch=1'i de
            # denemeden pes etmemek lazım — VRAM çok darsa gerçek taban bu olabilir.
            if bs > 1:
                if _fits(1):
                    lo, hi = 1, bs
                else:
                    self.tel.fatal("GPU batch=1 boyutunu bile sığdıramıyor!")
                    return -1
            else:
                self.tel.fatal("GPU batch=1 boyutunu bile sığdıramıyor!")
                return -1

        if hi is not None:
            # 2) BINARY SEARCH: lo (sığan) ile hi (sığmayan) arasında gerçek tavanı bul
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if _fits(mid):
                    lo = mid
                else:
                    hi = mid

        del initial_weights
        gc.collect()
        torch.cuda.empty_cache()

        self.tel.info(f"-> OPTIMAL MICRO-BATCH BULUNDU = {lo} (dinamik arama, tavan={hard_cap})")
        self.tel.record_metric("micro_batch", lo)
        return lo

    def build_and_validate(self, config: LlamaConfig, train_ds, collator,
                           seed_batch: int, use_grad_checkpointing: bool,
                           max_batch: Optional[int] = None) -> Tuple[torch.nn.Module, str, int]:
        self.tel.info("=" * 80)
        self.tel.info("BACKEND ORCHESTRATION & SMOKE TEST")
        self.tel.info("=" * 80)
        model, backend = self._select_backend(config, train_ds, collator)

        if use_grad_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            self.tel.info("-> Gradient Checkpointing batch araması öncesi AKTİF edildi.")

        opt_batch = self.find_optimal_batch(model, train_ds, collator, seed_batch=seed_batch, max_batch=max_batch)
        self.tel.record_metric("active_backend", backend)
        return model, backend, opt_batch

# ==============================================================================
# MODULE 6: CONFIGURATION MANAGER
# ==============================================================================
class ConfigurationManager:
    def __init__(self, telemetry: TelemetryLogger):
        self.tel = telemetry

    def generate_signatures(self, arch_dict: Dict[str, Any], policy_dict: Dict[str, Any]) -> Tuple[str, str]:
        self.tel.info("=" * 80)
        self.tel.info("CONFIGURATION FINGERPRINTING (SPLIT SIGNATURE)")
        self.tel.info("=" * 80)
        arch_sig = hashlib.sha256(
            json.dumps(arch_dict, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:16]
        policy_sig = hashlib.sha256(
            json.dumps(policy_dict, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:16]
        self.tel.info(f"-> Architecture Hash (IMMUTABLE): {arch_sig}")
        self.tel.info(f"-> Policy Hash       (MUTABLE)  : {policy_sig}")
        self.tel.record_metric("arch_sig", arch_sig)
        self.tel.record_metric("policy_sig", policy_sig)
        return arch_sig, policy_sig

# ==============================================================================
# MODULE 7: ATOMIC DRIVE PUBLISHER
# ==============================================================================
class AtomicDrivePublisher(TrainerCallback):
    def __init__(self, drive_ckpt_dir: str, local_logs_dir: str, drive_logs_dir: str,
                 keep_last: int = 3, telemetry: Optional[TelemetryLogger] = None,
                 drive_sync_steps: int = 500):
        self.drive_dir = drive_ckpt_dir
        self.local_logs_dir = local_logs_dir
        self.drive_logs_dir = drive_logs_dir
        self.keep_last = keep_last
        self.tel = telemetry
        self.drive_sync_steps = drive_sync_steps  # Drive'a sadece bu adım katlarında yükle
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
        core_files = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
        for f in core_files:
            if not os.path.exists(os.path.join(path, f)):
                return False
        has_weights = (
            os.path.exists(os.path.join(path, "model.safetensors")) or
            os.path.exists(os.path.join(path, "pytorch_model.bin")) or
            len(glob.glob(os.path.join(path, "*.safetensors"))) > 0
        )
        return has_weights

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
                    f.write(json.dumps({"step": step, "ts": datetime.now().isoformat()}))

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
                time.sleep(4)

        self._log("error", f"[PUBLISHER] checkpoint-{step} yayınlanamadı: {last_err!r}")
        return False

    def on_save(self, args, state, control, **kwargs):
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return

        step = state.global_step

        # [HIZLI KAYIT] Drive I/O sadece drive_sync_steps katlarında yapılır.
        # Diğer adımlarda yerel checkpoint zaten kaydedildi, Drive'a yükleme atlanır.
        if step % self.drive_sync_steps != 0:
            self._log("info", f"[PUBLISHER] checkpoint-{step} yerel kaydedildi (Drive sync: adım {step + (self.drive_sync_steps - step % self.drive_sync_steps)})")
            return

        src = os.path.join(args.output_dir, f"checkpoint-{step}")
        dst = os.path.join(self.drive_dir, f"checkpoint-{step}")

        if not self._publish_checkpoint(src, dst, step):
            return

        self._sync_logs()
        self._log("info", f"[PUBLISHER] Log senkronu tamam (step {step}).")

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
        self._log("info", "[PUBLISHER] Final log senkronizasyonu tamamlandı (on_train_end).")

# ==============================================================================
# MODULE 8: RESUME ORCHESTRATOR
# ==============================================================================
class ResumeOrchestrator:
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
        has_weights = (
            os.path.exists(os.path.join(path, "model.safetensors")) or
            os.path.exists(os.path.join(path, "pytorch_model.bin")) or
            len(glob.glob(os.path.join(path, "*.safetensors"))) > 0
        )
        return has_weights

    def _is_valid_drive(self, path: str) -> bool:
        if not os.path.isdir(path):
            return False
        return os.path.exists(os.path.join(path, "_COMPLETE")) and os.path.exists(os.path.join(path, "trainer_state.json"))

    def find_resume_point(self) -> Optional[str]:
        self.tel.info("=" * 80)
        self.tel.info("RESUME ORCHESTRATION & CHECKPOINT SELECTION")
        self.tel.info("=" * 80)

        arch_sig_file = os.path.join(self.drive_dir, "arch.sig")
        local_sig_file = os.path.join(self.local_dir, "arch.sig")

        # [CROSS-VERSION DISCOVERY]: Eğer aktif Drive dizininde checkpoint yoksa,
        # Drive'daki diğer ckpt_turklm_* dizinlerini tara ve en güncel doğrulanmış checkpoint'i bağla.
        drive_ckpts = [p for p in glob.glob(os.path.join(self.drive_dir, "checkpoint-*")) if self._is_valid_drive(p)]
        local_ckpts = [p for p in glob.glob(os.path.join(self.local_dir, "checkpoint-*")) if self._is_valid_local(p)]

        if not drive_ckpts and not local_ckpts:
            base_parent = os.path.dirname(self.drive_dir)
            all_drive_dirs = glob.glob(os.path.join(base_parent, "ckpt_turklm_*"))
            found_candidates = []
            for d in all_drive_dirs:
                if os.path.abspath(d) == os.path.abspath(self.drive_dir):
                    continue
                valid_in_d = [p for p in glob.glob(os.path.join(d, "checkpoint-*")) if self._is_valid_drive(p)]
                for cp in valid_in_d:
                    found_candidates.append((self._get_step(cp), cp, d))

            if found_candidates:
                found_candidates.sort(key=lambda x: x[0], reverse=True)
                highest_step, best_cp_path, source_dir = found_candidates[0]
                self.tel.info(
                    f"-> [SMART DISCOVERY] Önceki koşudan ({os.path.basename(source_dir)}) doğrulanmış "
                    f"checkpoint bulundu: {os.path.basename(best_cp_path)} (Adım {highest_step}). Aktif koşuya bağlanıyor..."
                )
                target_drive_cp = os.path.join(self.drive_dir, os.path.basename(best_cp_path))
                if os.path.exists(target_drive_cp):
                    shutil.rmtree(target_drive_cp, ignore_errors=True)
                shutil.copytree(best_cp_path, target_drive_cp)
                drive_ckpts = [target_drive_cp]

        if os.path.exists(arch_sig_file):
            with open(arch_sig_file, "r") as f:
                saved = f.read().strip()
            if saved != self.arch_sig:
                self.tel.warning(
                    f"Mimari/Sürüm imzası güncellendi: Önceki={saved} -> Yeni={self.arch_sig}.\n"
                    f"Aynı mimari üzerinden checkpoint devamlılığı (resume) sağlanıyor."
                )
                with open(arch_sig_file, "w") as f:
                    f.write(self.arch_sig)
            else:
                self.tel.info("-> Mimari imzası doğrulandı (Drive).")
        else:
            for p in (arch_sig_file, local_sig_file):
                with open(p, "w") as f:
                    f.write(self.arch_sig)
            self.tel.info("-> Mimari imzası yazıldı.")

        best_local = max(local_ckpts, key=self._get_step) if local_ckpts else None
        best_drive = max(drive_ckpts, key=self._get_step) if drive_ckpts else None

        step_local = self._get_step(best_local)
        step_drive = self._get_step(best_drive)

        self.tel.info(f"-> En güncel Local checkpoint : {os.path.basename(best_local) if best_local else 'YOK'} (Adım {step_local})")
        self.tel.info(f"-> En güncel Drive checkpoint : {os.path.basename(best_drive) if best_drive else 'YOK'} (Adım {step_drive})")

        resume_target = None
        if step_drive > step_local and best_drive is not None:
            self.tel.info(f"-> Drive daha güncel ({step_drive} > {step_local}). Smart Fetch devreye giriyor...")
            target = os.path.join(self.local_dir, os.path.basename(best_drive))
            if os.path.exists(target):
                shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(best_drive, target)
            self.tel.info("-> Smart Fetch tamamlandı.")
            resume_target = target
        elif best_local is not None:
            self.tel.info(f"-> Yerel checkpoint kullanılıyor: {os.path.basename(best_local)}")
            resume_target = best_local

        if resume_target is not None:
            state_file = os.path.join(resume_target, "trainer_state.json")
            if os.path.exists(state_file):
                try:
                    with open(state_file, "r") as f:
                        t_state = json.load(f)
                    best_ckpt_path = t_state.get("best_model_checkpoint")
                    if best_ckpt_path:
                        best_ckpt_name = os.path.basename(best_ckpt_path)
                        local_best_path = os.path.join(self.local_dir, best_ckpt_name)
                        drive_best_path = os.path.join(self.drive_dir, best_ckpt_name)
                        if not os.path.exists(local_best_path) and self._is_valid_drive(drive_best_path):
                            self.tel.info(f"-> [DUAL-FETCH] En iyi model ({best_ckpt_name}) Drive'dan yerel diske getiriliyor...")
                            shutil.copytree(drive_best_path, local_best_path)
                            self.tel.info(f"-> [DUAL-FETCH] En iyi model güvenceye alındı.")
                except Exception as dual_err:
                    self.tel.warning(f"Dual-fetch kontrol uyarısı: {dual_err!r}")

        return resume_target

# ==============================================================================
# MODULE 9: MODEL EVALUATOR
# ==============================================================================
class ModelEvaluator:
    def __init__(self, telemetry: TelemetryLogger):
        self.tel = telemetry
        self._FLASH_KEYWORDS = (
            "flash", "flashattention", "no available kernel",
            "unsupported by flashattention", "fa2",
        )

    def _is_flash_error(self, err: Exception) -> bool:
        msg = repr(err).lower()
        return any(k in msg for k in self._FLASH_KEYWORDS)

    def evaluate(self, trainer: Trainer, eval_ds) -> Tuple[float, float]:
        self.tel.info("=" * 80)
        self.tel.info("FULL HELD-OUT EVALUATION (TRUE PERPLEXITY)")
        self.tel.info("=" * 80)
        metrics = trainer.evaluate(eval_dataset=eval_ds)
        loss = metrics.get("eval_loss", float("nan"))
        try:
            ppl = math.exp(loss)
        except (OverflowError, ValueError):
            ppl = float("inf")
        self.tel.info(f"-> Eval Loss       : {loss:.4f}")
        self.tel.info(f"-> True Perplexity : {ppl:.2f}")
        self.tel.info(f"-> Eval Blok Sayısı: {len(eval_ds):,}")
        self.tel.record_metric("eval_loss", loss)
        self.tel.record_metric("eval_ppl", ppl)
        return loss, ppl

    def inference_smoke_test(self, model_path: str, tokenizer: PreTrainedTokenizerFast,
                             attn_impl: str, dtype: torch.dtype):
        self.tel.info("=" * 80)
        self.tel.info("INFERENCE SMOKE TEST")
        self.tel.info("=" * 80)
        device = torch.device("cuda")

        try:
            model = LlamaForCausalLM.from_pretrained(
                model_path, attn_implementation=attn_impl, **dtype_kwarg(dtype)
            ).to(device)
        except Exception as e:
            if attn_impl == "flash_attention_2" and self._is_flash_error(e):
                self.tel.warning(f"Flash Attention inference yüklenemedi: {e!r}. SDPA deneniyor...")
                model = LlamaForCausalLM.from_pretrained(
                    model_path, attn_implementation="sdpa", **dtype_kwarg(dtype)
                ).to(device)
            else:
                self.tel.fatal(f"Model yükleme hatası: {e!r}")
                return

        model.eval()
        prompts = [
            "Türkiye Cumhuriyeti'nin başkenti",
            "Mustafa Kemal Atatürk,",
            "İstanbul Boğazı,",
            "Yapay zekâ teknolojileri",
        ]
        for p in prompts:
            inputs = tokenizer(p, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs, max_new_tokens=40, do_sample=True,
                    top_k=40, top_p=0.9, temperature=0.7,
                    repetition_penalty=1.15,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            self.tel.info(f"\n[Prompt]: {p}\n[Model] : {text}")

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    # ---- 0. ENVIRONMENT ----
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    os.makedirs(LOCAL_LOGS_ROOT, exist_ok=True)

    boot_tel = TelemetryLogger(LOCAL_LOGS_ROOT, "bootstrap")

    # ---- 1. HARDWARE AUDIT ----
    specs = HardwareAuditor(boot_tel).audit()

    # ---- 2. HARDWARE PROFILES ----
    # NOT: Micro-batch artık burada TAHMİN EDİLMİYOR. Mimari (kaç layer/head/hidden)
    # hâlâ VRAM tavanına göre kademeli seçiliyor (bu bir model-boyutu kararı), ama
    # micro-batch dinamik arama (find_optimal_batch) ile her GPU için ayrı ayrı,
    # gerçek zamanlı ölçülerek bulunuyor — SEED_MICRO_BATCH sadece arama başlangıcı.
    if specs["vram_gb"] >= 35.0:
        PROFILE = "A100_HIGH"
        CONTEXT_LEN, HIDDEN_SIZE, INTERMEDIATE_SIZE = 2048, 1024, 4096
        NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS = 24, 16, 8
        USE_GRAD_CHECKPOINTING = False  # Liger sayesinde GC kapalı tam hız
        LR = 2.5e-4
    elif specs["vram_gb"] >= 20.0:
        PROFILE = "L4_MID"
        CONTEXT_LEN, HIDDEN_SIZE, INTERMEDIATE_SIZE = 2048, 1024, 4096
        NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS = 24, 16, 8
        USE_GRAD_CHECKPOINTING = True
        LR = 2.5e-4
    else:
        PROFILE = "T4_BUDGET"
        CONTEXT_LEN, HIDDEN_SIZE, INTERMEDIATE_SIZE = 1024, 768, 2048
        NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS = 12, 12, 12
        USE_GRAD_CHECKPOINTING = True
        LR = 3.0e-4

    # ---- HYPERPARAMETERS ----
    MAX_STEPS = 5000
    SAVE_STEPS = 100          # [HIZLI] Yerel checkpoint her 100 adımda (hızlı, saniyelik)
    DRIVE_SYNC_STEPS = 500    # [GÜVENLİ] Drive'a her 500 adımda yükle (ağır I/O sadece burada)
    EVAL_STEPS = 1000         # [TURBO] Eval yarıya düştü → her 1000 adımda bir
    LOGGING_STEPS = 25
    SEED_MICRO_BATCH = 2       # Dinamik aramanın başlangıç noktası (donanımdan bağımsız, hep küçük)
    # NOT: Bu artık batch aramasına bir TAVAN koymuyor (arama sadece %92 VRAM
    # sınırıyla duruyor — bkz. build_and_validate çağrısındaki max_batch=None).
    # Sadece accumulation hesabında referans: micro_batch bunu geçerse
    # (örn. A100'de 200 bulunursa) accum=1 olur, efektif batch micro_batch'e
    # eşitlenir — yani efektif batch artık 128'in ÜSTÜNE çıkabilir. Bu daha
    # fazla VRAM/throughput kullanımı demek, ama LR (aşağıda) hâlâ 128'lik
    # efektif batch'e göre ayarlanmıştı; efektif batch çok büyürse (ör. 2-3x)
    # LR'yi de orantılı artırmak (linear scaling) yakınsamayı hızlandırabilir.
    TARGET_EFFECTIVE_BATCH = 128

    # ---- 3. TOKENIZER ----
    tok_path = os.path.join(DRIVE_BASE, "tokenizer", "tokenizer.json")
    tokenizer, tok_hash = TokenizerAuditor(tok_path, boot_tel).audit_and_load()

    # ---- 4. DATASETS ----
    ds_mgr = DatasetManager(DRIVE_BASE, LOCAL_BASE, DATA_VERSION, CONTEXT_LEN, tok_hash, boot_tel)
    train_ds, eval_ds = ds_mgr.load_datasets()

    # ---- 5. SIGNATURES ----
    # ÖNEMLİ: run_version arch_dict'te DEĞİL, policy_dict'te.
    # arch_dict sadece mimariyi tanımlayan değerleri içermeli; yoksa her versiyon
    # atlamasında arch_sig değişir ve checkpoint-500 gibi kayıtlı koşulara resume edilemez.
    arch_dict = {
        "data_version": DATA_VERSION,
        "tok_hash": tok_hash, "vocab": len(tokenizer),
        "hidden": HIDDEN_SIZE, "intermediate": INTERMEDIATE_SIZE,
        "layers": NUM_LAYERS, "heads": NUM_HEADS, "kv_heads": NUM_KV_HEADS,
        "context": CONTEXT_LEN, "bf16": specs["use_bf16"],
    }
    policy_dict = {
        "run_version": RUN_VERSION,
        "max_steps": MAX_STEPS, "save_steps": SAVE_STEPS,
        "eval_steps": EVAL_STEPS, "logging_steps": LOGGING_STEPS,
        "lr": LR, "optimizer": "adamw", "scheduler": "cosine",
    }
    arch_sig, policy_sig = ConfigurationManager(boot_tel).generate_signatures(arch_dict, policy_dict)

    # ---- 6. PATHS ----
    RUN_NAME = f"turklm_{RUN_VERSION}_{PROFILE.lower()}_{arch_sig}"
    tel = TelemetryLogger(LOCAL_LOGS_ROOT, RUN_NAME)
    tel.info(f"AKTİF KOŞU ADI: {RUN_NAME}")

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

    # ---- 7. MODEL CONFIG & BACKEND SETUP ----
    # [TURBO] Liger Kernel durumu (dosyanın en başında monkey-patch edilmiştir)
    if _LIGER_ACTIVE:
        tel.info("--> [TURBO] Liger Kernel aktif: Fused RMSNorm + SwiGLU + CrossEntropy devrede.")
    else:
        tel.warning("--> [TURBO] Liger Kernel aktif değil (standart PyTorch katmanları kullanılıyor).")

    config = LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE, num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS, num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=CONTEXT_LEN, tie_word_embeddings=True,
        rope_theta=10000.0, rms_norm_eps=1e-5,
        bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, use_cache=False,
    )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    backend = BackendOrchestrator(specs, tel)

    model, active_backend, micro_batch = backend.build_and_validate(
        config, train_ds, collator,
        seed_batch=SEED_MICRO_BATCH,
        use_grad_checkpointing=USE_GRAD_CHECKPOINTING,
        max_batch=None,  # Artık 128'de tavan yok: arama sadece %92 VRAM
                         # güvenlik sınırıyla (ve dataset boyutuyla) durur —
                         # A100'de kalan VRAM'in tamamına yakınını kullanır.
    )

    # [TURBO] Dynamo hata toleransı: Trace/compile hatalarında sessizce eager moda düşer
    if hasattr(torch, "_dynamo"):
        try:
            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass

    param_count = sum(p.numel() for p in model.parameters())
    tel.info(f"-> Toplam Parametre Sayısı: {param_count / 1e6:.2f} M")
    tel.record_metric("param_count_M", param_count / 1e6)

    if model.config.vocab_size != len(tokenizer):
        tel.warning(f"Vocab mismatch: {model.config.vocab_size} -> {len(tokenizer)}. Düzeltiliyor.")
        model.resize_token_embeddings(len(tokenizer))

    # ---- 8. CHINCHILLA RAPORU & ACCUMULATION ----
    accum = max(1, TARGET_EFFECTIVE_BATCH // micro_batch)
    effective_batch = micro_batch * accum
    tokens_per_step = effective_batch * CONTEXT_LEN
    total_tokens = MAX_STEPS * tokens_per_step
    chinchilla_ratio = total_tokens / param_count

    tel.info("=" * 80)
    tel.info("CHINCHILLA HESAPLAMA & BATCH RAPORU")
    tel.info("=" * 80)
    tel.info(f"-> Micro-Batch / Accum     : {micro_batch} / {accum} (Efektif: {effective_batch})")
    if effective_batch != TARGET_EFFECTIVE_BATCH:
        tel.warning(f"-> Efektif Batch ({effective_batch}) hedef ({TARGET_EFFECTIVE_BATCH}) ile tam örtüşmüyor (kabul edilebilir sapma).")

    tel.info(f"-> Adım Başına Token       : {tokens_per_step:,}")
    tel.info(f"-> Hedef Toplam Token      : {total_tokens:,}")
    tel.info(f"-> Token / Parametre Oranı : {chinchilla_ratio:.3f}")
    if chinchilla_ratio < 5.0:
        tel.warning("DİKKAT: Token bütçesi Chinchilla-optimal (<20.0) altında; compute-limited eğitim.")
    tel.record_metric("tokens_per_step", tokens_per_step)
    tel.record_metric("total_tokens", total_tokens)
    tel.record_metric("chinchilla_ratio", chinchilla_ratio)

    # ---- 9. RESUME NOKTASI ----
    last_checkpoint = ResumeOrchestrator(LOCAL_CKPT, DRIVE_CKPT, arch_sig, tel).find_resume_point()

    # ---- 10. TRAINING ARGUMENTS ----
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": LOCAL_CKPT,
        "overwrite_output_dir": False,
        "max_steps": MAX_STEPS,
        "per_device_train_batch_size": micro_batch,
        "per_device_eval_batch_size": min(micro_batch, 8),
        "gradient_accumulation_steps": accum,
        "learning_rate": LR,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "adam_beta1": 0.9, "adam_beta2": 0.95, "adam_epsilon": 1e-8,
        "bf16": specs["use_bf16"], "fp16": specs["use_fp16"], "tf32": specs["use_tf32"],
        "gradient_checkpointing": USE_GRAD_CHECKPOINTING,
        "save_steps": SAVE_STEPS, "eval_steps": EVAL_STEPS,
        "save_total_limit": 10,              # Yerel: 10 checkpoint tutar (100 adım aralıklı = son 1000 adım)
        "load_best_model_at_end": False,     # [TURBO] save_steps≠eval_steps olduğu için kapalı
        "prediction_loss_only": True,
        "dataloader_num_workers": min(4, os.cpu_count() or 2),
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": True,
        "dataloader_prefetch_factor": 4,   # [TURBO] GPU boşta beklemesini azaltır
        "remove_unused_columns": False,    # [KRİTİK] input_ids'in filtrelenmesini engeller
        "torch_compile": False,            # [LIGER UYUMLU] Liger Triton çekirdekleri ile çakışmayı önler
        "logging_steps": LOGGING_STEPS, "logging_first_step": True,
        "logging_dir": LOCAL_LOGS,
        "seed": 42, "data_seed": 42,
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

    # ---- 11. TRAINER BINDING ----
    publisher = AtomicDrivePublisher(
        drive_ckpt_dir=DRIVE_CKPT,
        local_logs_dir=LOCAL_LOGS,
        drive_logs_dir=DRIVE_LOGS,
        keep_last=3,
        telemetry=tel,
        drive_sync_steps=DRIVE_SYNC_STEPS,
    )
    trainer_kwargs = {
        "model": model, "args": args,
        "train_dataset": train_ds, "eval_dataset": eval_ds,
        "data_collator": collator, "callbacks": [publisher],
    }
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    # ---- 12. PRE-FLIGHT CHECKLIST ----
    tel.info("=" * 80)
    tel.info("PRE-FLIGHT DOĞRULAMA (v10.0 TURBO)")
    tel.info("=" * 80)
    tel.info("1. Tokenizer sözleşmesi             : ONAYLANDI")
    tel.info("2. Cache bütünlüğü (_COMPLETE)      : ONAYLANDI")
    tel.info(f"3. Dikkat mekanizması               : {active_backend.upper()}")
    tel.info(f"4. Güvenli Dinamik Micro-Batch      : {micro_batch}")
    tel.info(f"5. Efektif Batch                    : {effective_batch}")
    tel.info(f"6. Gradient Checkpointing           : {USE_GRAD_CHECKPOINTING}")
    tel.info(f"7. Mimari İmzası (IMMUTABLE)        : {arch_sig}")
    tel.info(f"8. Politika İmzası (MUTABLE)        : {policy_sig}")
    tel.info(f"9. Çalışma Durumu                   : {'RESUME' if last_checkpoint else 'FRESH'}")
    tel.info(f"10. Toplam Adım (MAX_STEPS)         : {MAX_STEPS:,}")
    tel.info(f"11. [TURBO] Liger Kernel            : {'AKTİF' if _LIGER_ACTIVE else 'KAPALI (pip install liger-kernel)'}")
    tel.info(f"12. [TURBO] torch.compile           : {'AKTİF' if kwargs.get('torch_compile') else 'KAPALI (Liger Kernel devrede, çakışmasız tam hız)'}")
    tel.info(f"13. [TURBO] Dataloader Prefetch     : 4")
    tel.info("=" * 80)
    tel.info("EĞİTİM BAŞLIYOR...")

    # Caching allocator'ı sıfırla, Trainer temiz havuzla başlasın
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # ---- 13. LIFTOFF ----
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # ---- 14. FULL HELD-OUT EVALUATION ----
    evaluator = ModelEvaluator(tel)
    eval_loss, eval_ppl = evaluator.evaluate(trainer, eval_ds)

    # ---- 15. ATOMIC FINAL MODEL SAVE ----
    tel.info("=" * 80)
    tel.info("FINAL MODEL KAYDI (ATOMIC)")
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

    manifest = {
        "version": RUN_VERSION, "profile": PROFILE,
        "arch_sig": arch_sig, "policy_sig": policy_sig,
        "arch": arch_dict, "policy": policy_dict,
        "compute": {
            "micro_batch": micro_batch, "accum": accum,
            "effective_batch": effective_batch,
            "tokens_per_step": tokens_per_step,
            "total_tokens": total_tokens,
            "chinchilla_ratio": chinchilla_ratio,
        },
        "final_metrics": {
            **getattr(train_result, "metrics", {}),
            "eval_loss_heldout": eval_loss,
            "perplexity_heldout": eval_ppl,
        },
        "gpu": specs["gpu_name"],
        "active_backend": active_backend,
        "param_count": param_count,
    }
    with open(os.path.join(LOCAL_FINAL, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(LOCAL_FINAL, "_COMPLETE"), "w") as f:
        f.write(f"arch_sig={arch_sig}\npolicy_sig={policy_sig}\n")

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
    tel.info(f"-> Nihai model Drive'a kaydedildi: {DRIVE_FINAL}")

    tel.save_manifest(os.path.join(DRIVE_FINAL, "telemetry_manifest.json"))

    # ---- 16. FINAL LOG SYNC ----
    tel.info("-> Son loglar Drive'a aktarılıyor...")
    try:
        shutil.copytree(LOCAL_LOGS, DRIVE_LOGS, dirs_exist_ok=True)
        if os.path.exists(LOCAL_LOGS_ROOT):
            for fname in os.listdir(LOCAL_LOGS_ROOT):
                src = os.path.join(LOCAL_LOGS_ROOT, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(DRIVE_LOGS, fname))
        tel.info(f"-> Tüm loglar kalıcı Drive klasöründe: {DRIVE_LOGS}")
    except Exception as e:
        tel.warning(f"Nihai log sync hatası: {e!r}")

    # ---- 17. MEMORY CLEANUP ----
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ---- 18. INFERENCE DEMO ----
    dtype = torch.bfloat16 if specs["use_bf16"] else torch.float16
    test_model_path = LOCAL_FINAL if os.path.exists(LOCAL_FINAL) else DRIVE_FINAL
    evaluator.inference_smoke_test(test_model_path, tokenizer, active_backend, dtype)

    # ---- 19. SHUTDOWN ----
    tel.info("=" * 80)
    tel.info(f"TURKLM {RUN_VERSION} — TÜM OPERASYON BAŞARIYLA TAMAMLANDI")
    tel.info(f"Model Konumu   : {DRIVE_FINAL}")
    tel.info(f"Held-out PPL   : {eval_ppl:.2f}")
    tel.info(f"Toplam Token   : {total_tokens:,}")
    tel.info("=" * 80)


if __name__ == "__main__":
    main()