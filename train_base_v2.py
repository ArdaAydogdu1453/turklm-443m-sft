# ==============================================================================
# TURKLM v11.0_base_v2 - PRODUCTION MASTER (3-STAGE NANO-GPT CURRICULUM PRETRAINING)
# ==============================================================================
# ==============================================================================
# Bu kod, TurkLM Base-v2 mimarisi için musabc/nanogpt-tr-v5-data (~15 Milyar token)
# 3-aşamalı (Curriculum Learning: Web -> Medium -> Premium) ön eğitim pipeline'ıdır.
#
# Kritik Mimari İlkeleri:
#   1. Veri: musabc/nanogpt-tr-v5-data (v5_stage1.bin, v5_stage2.bin, v5_stage3.bin, v5_val.bin)
#   2. Tokenizer: tokenizer-tr-v5.json (32,000 BPE Vocab, nanogpt uyumlu)
#   3. Süreklilik (Curriculum): Stage 1 -> Stage 2 -> Stage 3 kesintisiz ağırlık aktarımı.
#   4. Çok Seviyeli Resume: Her aşamanın kendi checkpoint'leri Drive ile atomik senkronize edilir.
#   5. Lazy np.memmap: Çoklu worker süreçlerinde (DataLoader) fork kilitlenmelerini önleyen mimari.
#   6. Hız Optimizasyonu: Liger Kernel + Flash Attention 2 / SDPA + adamw_torch_fused + BF16/TF32.
#   7. Val Koruması: 150M tokenlık dev val seti yerine hızlı ve istatistiksel 2M token subsample.
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
import numpy as np

try:
    import torch._dynamo
except ImportError:
    pass

# ==============================================================================
# FLASH ATTENTION 2 DENETİMİ (OTOMATİK ALGILAMA)
# ==============================================================================
def _init_flash_attention() -> bool:
    try:
        import flash_attn  # noqa: F401
        print("[FLASH-ATTN] Kurulu ve aktif, kullanılacak.")
        return True
    except ImportError:
        print("[FLASH-ATTN] Kurulu değil -> SDPA (Scaled Dot-Product Attention) kullanılacak.")
        return False
    except Exception as e:
        print(f"[FLASH-ATTN] Kontrol hatası: {e!r} -> SDPA fallback.")
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
        print("[LIGER] Aktif - Fused CrossEntropy + SwiGLU + RMSNorm + RoPE devrede.")
    except ImportError:
        print("[LIGER] Kurulu değil -> Standart PyTorch katmanları devrede.")
        _LIGER_ACTIVE = False
    except Exception as e:
        print(f"[LIGER] Aktivasyon hatası: {e!r}")
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
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerFast,
    default_data_collator
)

# ==============================================================================
# MODULE 0: GLOBAL CONSTANTS & PIPELINE CONFIGURATION
# ==============================================================================
RUN_VERSION = "v11.0_base_v2"
DRIVE_BASE = "/content/drive/MyDrive/Turkce_Tiny_LM" if _IN_COLAB else os.path.abspath("./drive_turklm")
LOCAL_BASE = "/content" if _IN_COLAB else os.path.abspath("./local_workspace")
LOCAL_LOGS_ROOT = os.path.join(LOCAL_BASE, "logs_telemetry")

# nanogpt-tr-v5 veri seti yerel veya Drive dizini
DATA_DIR_LOCAL = "/content/nanogpt_tr_v5_data" if _IN_COLAB else os.path.abspath("./nanogpt_tr_v5_data")
DATA_DIR_DRIVE = os.path.join(DRIVE_BASE, "nanogpt_tr_v5_data")
HF_DATASET_REPO = "musabc/nanogpt-tr-v5-data"

# Optimizer Tipi: PyTorch yerel CUDA Fused AdamW
OPTIMIZER_TYPE = "adamw_torch_fused"

# ==============================================================================
# 3-AŞAMALI EĞİTİM (CURRICULUM) VE HİPERPARAMETRE TANIMLARI
# ==============================================================================
# max_steps: -1 verilirse tüm epoch (~15 milyar token, ~76 saat) eğitilir.
# Colab bütçesi için önerilen: 3000 / 3000 / 2000 (~12-14 saat A100, ~2 Milyar token, Chinchilla %30+).
STAGE_SPECS = [
    {
        "id": "stage1",
        "file": "v5_stage1.bin",
        "name": "Stage 1 - Web Tier (OSCAR, mC4, Forum, FineWeb-HQ)",
        "tokens_approx": "2.94 Milyar",
        "learning_rate": 2.5e-4,
        "weight_decay": 0.1,
        "epochs": 1,
        "max_steps": 3000,     # ~4-5 saat A100 (tam veri için -1 yapabilirsiniz)
        "warmup_ratio": 0.03,
    },
    {
        "id": "stage2",
        "file": "v5_stage2.bin",
        "name": "Stage 2 - Medium Tier (BellaTurca, Cosmos, CulturaX, Cosmopedia)",
        "tokens_approx": "9.03 Milyar",
        "learning_rate": 1.8e-4,
        "weight_decay": 0.1,
        "epochs": 1,
        "max_steps": 3000,     # ~4-5 saat A100 (tam veri için -1 yapabilirsiniz)
        "warmup_ratio": 0.02,
    },
    {
        "id": "stage3",
        "file": "v5_stage3.bin",
        "name": "Stage 3 - Premium Tier (Wiki, Wikisource, Akademik, FinePDFs)",
        "tokens_approx": "2.97 Milyar",
        "learning_rate": 8.0e-5,
        "weight_decay": 0.1,
        "epochs": 1,
        "max_steps": 2000,     # ~2.5-3 saat A100 (tam veri için -1 yapabilirsiniz)
        "warmup_ratio": 0.02,
    },
]

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
        self.tel.info("HARDWARE & ENVIRONMENT AUDIT (BASE-V2)")
        self.tel.info("=" * 80)

        if not torch.cuda.is_available():
            self.tel.fatal("CUDA GPU bulunamadı! Runtime > Change runtime type > GPU (A100/L4/T4) seçin.")

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
                    f"   Lütfen Colab menüsünden: 'Runtime -> Restart Session' yapın!"
                )
        except Exception:
            pass

        self.tel.info("HARDWARE AUDIT PASSED.")
        return specs

# ==============================================================================
# MODULE 3: TOKENIZER AUDITOR (NANOGPT V5 32K BPE)
# ==============================================================================
class TokenizerAuditor:
    def __init__(self, tokenizer_path: str, telemetry: TelemetryLogger):
        self.path = tokenizer_path
        self.tel = telemetry

    def audit_and_load(self) -> Tuple[PreTrainedTokenizerFast, str]:
        self.tel.info("=" * 80)
        self.tel.info("TOKENIZER IMMUTABILITY AUDIT (NANOGPT V5 BPE)")
        self.tel.info("=" * 80)

        if not os.path.exists(self.path):
            self.tel.fatal(f"Tokenizer dosyası bulunamadı: {self.path}")

        with open(self.path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:12]
        self.tel.info(f"-> Tokenizer Dosya Yolu: {self.path}")
        self.tel.info(f"-> Tokenizer SHA-256    : {file_hash}")
        self.tel.record_metric("tokenizer_hash", file_hash)

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self.path,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )

        # Pad token yoksa EOS'u pad yap (Causal LM standardı)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
                self.tel.info(f"-> pad_token tanımlı değildi, eos_token ({tokenizer.eos_token}) pad olarak atandı.")
            else:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})

        vocab_size = len(tokenizer)
        self.tel.info(f"-> Toplam Vocab size: {vocab_size} (Beklenen: 32000)")
        self.tel.info(f"-> BOS: {tokenizer.bos_token} (ID {tokenizer.bos_token_id})")
        self.tel.info(f"-> EOS: {tokenizer.eos_token} (ID {tokenizer.eos_token_id})")
        self.tel.info(f"-> PAD: {tokenizer.pad_token} (ID {tokenizer.pad_token_id})")
        self.tel.record_metric("vocab_size", vocab_size)
        self.tel.info("TOKENIZER AUDIT PASSED.")
        return tokenizer, file_hash

# ==============================================================================
# MODULE 4: HIGH-PERFORMANCE LAZY MEMMAP DATASET & MANAGER
# ==============================================================================
class NanoGPTMemmapDataset(torch.utils.data.Dataset):
    """
    musabc/nanogpt-tr-v5-data uint16 binary dosyalarını sıfır bellek yüküyle okuyan
    ve multi-worker DataLoader fork kilitlenmelerini önlemek için lazy memmap kullanan Dataset.
    """
    def __init__(self, bin_path: str, block_size: int, max_samples: Optional[int] = None):
        self.bin_path = bin_path
        self.block_size = block_size
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary veri dosyası bulunamadı: {bin_path}")

        file_size_bytes = os.path.getsize(bin_path)
        total_tokens = file_size_bytes // 2  # np.uint16 = 2 bytes
        self.total_blocks = total_tokens // block_size

        if max_samples is not None and max_samples > 0:
            self.total_blocks = min(self.total_blocks, max_samples)

        self.data: Optional[np.memmap] = None

    def _ensure_open(self):
        # Her worker process kendi handle'ını açar (thread-safe / process-safe)
        if self.data is None:
            self.data = np.memmap(self.bin_path, dtype=np.uint16, mode='r')

    def __len__(self) -> int:
        return self.total_blocks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._ensure_open()
        start = idx * self.block_size
        end = start + self.block_size
        chunk = self.data[start:end].astype(np.int64)
        input_ids = torch.from_numpy(chunk)
        # Standart Causal LM pretraining: labels input_ids ile birebir aynıdır
        return {"input_ids": input_ids, "labels": input_ids.clone()}


class NanoGPTDatasetManager:
    """
    Veri setini doğrular, gerekirse HuggingFace'den otomatik indirir ve
    aşamalara göre NanoGPTMemmapDataset döndürür.
    """
    def __init__(self, local_dir: str, drive_dir: str, repo_id: str, telemetry: TelemetryLogger):
        self.local_dir = local_dir
        self.drive_dir = drive_dir
        self.repo_id = repo_id
        self.tel = telemetry

    def ensure_data_ready(self) -> str:
        self.tel.info("=" * 80)
        self.tel.info("NANOGPT-TR-V5 DATASET DISCOVERY & SYNC")
        self.tel.info("=" * 80)

        needed_files = ["v5_stage1.bin", "v5_stage2.bin", "v5_stage3.bin", "v5_val.bin", "tokenizer-tr-v5.json"]

        # 1. Local NVMe kontrolü (en hızlısı)
        if os.path.exists(self.local_dir):
            if all(os.path.exists(os.path.join(self.local_dir, f)) for f in needed_files):
                self.tel.info(f"-> Yerel NVMe veri seti bulundu ve doğrulandı: {self.local_dir}")
                return self.local_dir

        # 2. Drive kontrolü
        if os.path.exists(self.drive_dir):
            if all(os.path.exists(os.path.join(self.drive_dir, f)) for f in needed_files):
                self.tel.info(f"-> Drive'da doğrulanmış veri seti bulundu: {self.drive_dir}")
                self.tel.info(f"-> NVMe hız avantajı için yerel diske kopyalanıyor ({self.local_dir})...")
                os.makedirs(self.local_dir, exist_ok=True)
                for f in needed_files:
                    src_f = os.path.join(self.drive_dir, f)
                    dst_f = os.path.join(self.local_dir, f)
                    if not os.path.exists(dst_f):
                        shutil.copy2(src_f, dst_f)
                self.tel.info("-> NVMe kopyalama tamamlandı.")
                return self.local_dir

        # 3. Bulunamadıysa HuggingFace snapshot_download
        self.tel.info(f"-> Veri seti ne yerel ne Drive'da bulunamadı.")
        self.tel.info(f"-> HuggingFace Hub'dan otomatik indiriliyor: '{self.repo_id}' (~30.9 GB)...")
        os.makedirs(self.local_dir, exist_ok=True)

        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                local_dir=self.local_dir,
                local_dir_use_symlinks=False,
            )
            self.tel.info("-> HuggingFace indirmesi başarıyla tamamlandı.")
        except Exception as e:
            self.tel.fatal(
                f"HuggingFace veri seti indirilemedi: {e!r}\n"
                f"Lütfen manuel olarak 'huggingface-cli download {self.repo_id} --local-dir {self.local_dir}' çalıştırın."
            )

        # Doğrulama
        missing = [f for f in needed_files if not os.path.exists(os.path.join(self.local_dir, f))]
        if missing:
            self.tel.fatal(f"İndirme sonrası eksik dosyalar tespit edildi: {missing}")

        return self.local_dir

    def get_stage_dataset(self, data_root: str, stage_file: str, block_size: int,
                          max_samples: Optional[int] = None) -> NanoGPTMemmapDataset:
        file_path = os.path.join(data_root, stage_file)
        ds = NanoGPTMemmapDataset(file_path, block_size=block_size, max_samples=max_samples)
        self.tel.info(f"-> Yüklenen Dosya : {stage_file}")
        self.tel.info(f"   Toplam Blok   : {len(ds):,} blok ({len(ds)*block_size:,} token)")
        return ds

    def get_validation_dataset(self, data_root: str, block_size: int,
                              max_val_samples: int = 1500) -> NanoGPTMemmapDataset:
        val_path = os.path.join(data_root, "v5_val.bin")
        # 150M tokenlık devasa val kümesi yerine her eval adımında 1500 blok (~3M token) yeterlidir
        val_ds = NanoGPTMemmapDataset(val_path, block_size=block_size, max_samples=max_val_samples)
        self.tel.info(f"-> Hızlı Validation Seti: {len(val_ds):,} blok (~{len(val_ds)*block_size/1e6:.1f} M token)")
        return val_ds

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
                           seed_batch: int = 4, max_batch: Optional[int] = None) -> int:
        self.tel.info("=" * 80)
        self.tel.info("DİNAMİK MICRO-BATCH ARAMASI (VRAM GÜVENLİK TAVANI %92)")
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
                time.sleep(0.2)
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

        # Exponential Ramp-Up (2x)
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
                self.tel.fatal("GPU batch=1 boyutunu bile sığdıramıyor!")
                return -1

        if hi is not None:
            # Binary search
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if _fits(mid):
                    lo = mid
                else:
                    hi = mid

        del initial_weights
        gc.collect()
        torch.cuda.empty_cache()

        self.tel.info(f"-> OPTİMAL MICRO-BATCH BULUNDU = {lo} (Tavan: {hard_cap})")
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
            self.tel.info("-> Gradient Checkpointing aktif edildi.")

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
# MODULE 7: STAGE-AWARE ATOMIC DRIVE PUBLISHER
# ==============================================================================
class StageAtomicDrivePublisher(TrainerCallback):
    """
    Drive FUSE takılmalarını önlemek için yerel NVMe'ye her save_steps'te kayıt yapar;
    ağır Drive aktarımını yalnızca drive_sync_steps katlarında ve atomik (.partial) yürütür.
    """
    def __init__(self, stage_id: str, drive_ckpt_dir: str, local_logs_dir: str,
                 drive_logs_dir: str, keep_last: int = 2, telemetry: Optional[TelemetryLogger] = None,
                 drive_sync_steps: int = 300):
        self.stage_id = stage_id
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
            getattr(self.tel, level)(f"[{self.stage_id.upper()}] {msg}")
        else:
            print(f"[{self.stage_id.upper()}] {msg}")

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
                    f.write(json.dumps({"stage": self.stage_id, "step": step, "ts": datetime.now().isoformat()}))

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
# MODULE 8: STAGE-AWARE RESUME & CURRICULUM ORCHESTRATOR
# ==============================================================================
class StageResumeOrchestrator:
    """
    Belirli bir stage için en güncel doğrulanmış checkpoint'i bulur.
    Eğer Drive'daki checkpoint yerelden daha güncelse NVMe'ye kopyalar (Smart Fetch).
    """
    def __init__(self, stage_id: str, local_dir: str, drive_dir: str,
                 arch_sig: str, telemetry: TelemetryLogger):
        self.stage_id = stage_id
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
        self.tel.info(f"[{self.stage_id.upper()}] RESUME ORCHESTRATION & CHECKPOINT SELECTION")
        self.tel.info("=" * 80)

        drive_ckpts = [p for p in glob.glob(os.path.join(self.drive_dir, "checkpoint-*")) if self._is_valid_drive(p)]
        local_ckpts = [p for p in glob.glob(os.path.join(self.local_dir, "checkpoint-*")) if self._is_valid_local(p)]

        best_local = max(local_ckpts, key=self._get_step) if local_ckpts else None
        best_drive = max(drive_ckpts, key=self._get_step) if drive_ckpts else None

        step_local = self._get_step(best_local)
        step_drive = self._get_step(best_drive)

        self.tel.info(f"-> En güncel Yerel checkpoint : {os.path.basename(best_local) if best_local else 'YOK'} (Adım {step_local})")
        self.tel.info(f"-> En güncel Drive checkpoint : {os.path.basename(best_drive) if best_drive else 'YOK'} (Adım {step_drive})")

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
# MODULE 9: MODEL EVALUATOR & BENCHMARK INFERENCE
# ==============================================================================
class ModelEvaluator:
    def __init__(self, telemetry: TelemetryLogger):
        self.tel = telemetry

    def evaluate(self, trainer: Trainer, eval_ds) -> Tuple[float, float]:
        self.tel.info("=" * 80)
        self.tel.info("HELD-OUT VALIDATION (NANOGPT V5 VAL SET - TRUE PERPLEXITY)")
        self.tel.info("=" * 80)
        metrics = trainer.evaluate(eval_dataset=eval_ds)
        loss = metrics.get("eval_loss", float("nan"))
        try:
            ppl = math.exp(loss)
        except (OverflowError, ValueError):
            ppl = float("inf")
        self.tel.info(f"-> Eval Loss       : {loss:.4f}")
        self.tel.info(f"-> True Perplexity : {ppl:.2f}")
        self.tel.info(f"-> Değerlendirilen Blok Sayısı: {len(eval_ds):,}")
        self.tel.record_metric("eval_loss", loss)
        self.tel.record_metric("eval_ppl", ppl)
        return loss, ppl

    def inference_smoke_test(self, model_path: str, tokenizer: PreTrainedTokenizerFast,
                             attn_impl: str, dtype: torch.dtype):
        self.tel.info("=" * 80)
        self.tel.info("BASE-V2 TURKISH GENERATION BENCHMARK TEST")
        self.tel.info("=" * 80)
        device = torch.device("cuda")

        try:
            model = LlamaForCausalLM.from_pretrained(
                model_path, attn_implementation=attn_impl, **dtype_kwarg(dtype)
            ).to(device)
        except Exception as e:
            self.tel.warning(f"Attn={attn_impl} yüklenemedi: {e!r}. SDPA deneniyor...")
            model = LlamaForCausalLM.from_pretrained(
                model_path, attn_implementation="sdpa", **dtype_kwarg(dtype)
            ).to(device)

        model.eval()
        prompts = [
            "Türkiye Cumhuriyeti'nin başkenti",
            "Mustafa Kemal Atatürk,",
            "İstanbul Boğazı, Asya ve Avrupa kıtalarını",
            "Yapay zekâ ve derin öğrenme algoritmaları,",
            "Türk edebiyatında klasik dönem",
            "Güneş Sistemi'nin en büyük gezegeni olan Jüpiter,"
        ]
        for p in prompts:
            inputs = tokenizer(p, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs, max_new_tokens=45, do_sample=True,
                    top_k=40, top_p=0.9, temperature=0.7,
                    repetition_penalty=1.15,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            self.tel.info(f"\n[Prompt]: {p}\n[Model] : {text}")

# ==============================================================================
# MODULE 10: MULTI-STAGE MASTER PIPELINE (LIFTOFF)
# ==============================================================================
def main():
    # ---- 0. ENVIRONMENT SETUP ----
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_HOME"] = os.path.join(DRIVE_BASE, "hf_cache")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if _IN_COLAB and not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")

    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    os.makedirs(LOCAL_LOGS_ROOT, exist_ok=True)
    os.makedirs(DRIVE_BASE, exist_ok=True)

    boot_tel = TelemetryLogger(LOCAL_LOGS_ROOT, "bootstrap_base_v2")

    # ---- 1. HARDWARE AUDIT ----
    specs = HardwareAuditor(boot_tel).audit()

    # ---- 2. HARDWARE PROFILES & ARCHITECTURE ----
    # 32K vocab BPE için optimize edilmiş LLaMA mimarisi
    if specs["vram_gb"] >= 35.0:
        PROFILE = "A100_HIGH"
        CONTEXT_LEN, HIDDEN_SIZE, INTERMEDIATE_SIZE = 2048, 1024, 4096
        NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS = 24, 16, 8
        USE_GRAD_CHECKPOINTING = False  # Liger ile tam hız
        TARGET_EFFECTIVE_BATCH = 120    # 120 * 2048 = 245,760 token / adım
        SEED_MICRO_BATCH = 8
    elif specs["vram_gb"] >= 20.0:
        PROFILE = "L4_MID"
        CONTEXT_LEN, HIDDEN_SIZE, INTERMEDIATE_SIZE = 2048, 1024, 4096
        NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS = 24, 16, 8
        USE_GRAD_CHECKPOINTING = True
        TARGET_EFFECTIVE_BATCH = 120
        SEED_MICRO_BATCH = 4
    else:
        PROFILE = "T4_BUDGET"
        CONTEXT_LEN, HIDDEN_SIZE, INTERMEDIATE_SIZE = 1024, 768, 2048
        NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS = 12, 12, 12
        USE_GRAD_CHECKPOINTING = True
        TARGET_EFFECTIVE_BATCH = 64
        SEED_MICRO_BATCH = 2

    # Genel Kayıt ve Senkron Aralıkları
    SAVE_STEPS = 100          # Yerel NVMe'ye her 100 adımda
    DRIVE_SYNC_STEPS = 300    # Ağır Google Drive senkronu her 300 adımda
    EVAL_STEPS = 500          # Her 500 adımda hızlı validation
    LOGGING_STEPS = 20

    # ---- 3. DATASET & TOKENIZER HAZIRLIĞI ----
    ds_manager = NanoGPTDatasetManager(DATA_DIR_LOCAL, DATA_DIR_DRIVE, HF_DATASET_REPO, boot_tel)
    ready_data_dir = ds_manager.ensure_data_ready()

    tok_file = os.path.join(ready_data_dir, "tokenizer-tr-v5.json")
    tokenizer, tok_hash = TokenizerAuditor(tok_file, boot_tel).audit_and_load()
    # DEĞİŞTİ: Tokenizer'da BOS/EOS/PAD/UNK special tokenları sonradan eklenmiş
    # Base vocab 32000, +4 special = 32004 normaldir.
    assert 32000 <= len(tokenizer) <= 32100, \
        f"Tokenizer vocab 32000-32100 arasında olmalı, {len(tokenizer)} bulundu!"

    # Tokenizer'ı kalıcı Drive dizinine de yedekle
    drive_tok_dir = os.path.join(DRIVE_BASE, "tokenizer_v5")
    os.makedirs(drive_tok_dir, exist_ok=True)
    tokenizer.save_pretrained(drive_tok_dir)

    # ---- 4. SIGNATURES & RUN PATHS ----
    arch_dict = {
        "model_name": "TurkLM_Base_v2",
        "data_repo": HF_DATASET_REPO,
        "tok_hash": tok_hash,
        "vocab": len(tokenizer),
        "hidden": HIDDEN_SIZE,
        "intermediate": INTERMEDIATE_SIZE,
        "layers": NUM_LAYERS,
        "heads": NUM_HEADS,
        "kv_heads": NUM_KV_HEADS,
        "context": CONTEXT_LEN,
        "bf16": specs["use_bf16"],
    }
    policy_dict = {
        "run_version": RUN_VERSION,
        "profile": PROFILE,
        "save_steps": SAVE_STEPS,
        "drive_sync_steps": DRIVE_SYNC_STEPS,
        "eval_steps": EVAL_STEPS,
        "optimizer": OPTIMIZER_TYPE,
    }
    arch_sig, policy_sig = ConfigurationManager(boot_tel).generate_signatures(arch_dict, policy_dict)

    RUN_NAME = f"turklm_base_v2_{PROFILE.lower()}_{arch_sig}"
    tel = TelemetryLogger(LOCAL_LOGS_ROOT, RUN_NAME)
    tel.info(f"AKTİF BASE-V2 KOŞU ADI: {RUN_NAME}")

    # ---- 5. MODEL MİMARİSİ VE İLK BAŞLATMA ----
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=CONTEXT_LEN,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )

    # Dataloader ve collator hazırlığı
    collator = default_data_collator
    backend = BackendOrchestrator(specs, tel)

    # Batch boyutu testi için Stage 1'den geçici küçük dataset
    dummy_train_ds = ds_manager.get_stage_dataset(ready_data_dir, "v5_stage1.bin", CONTEXT_LEN, max_samples=100)

    model, active_backend, micro_batch = backend.build_and_validate(
        config, dummy_train_ds, collator,
        seed_batch=SEED_MICRO_BATCH,
        use_grad_checkpointing=USE_GRAD_CHECKPOINTING,
        max_batch=None
    )
    del dummy_train_ds

    param_count = sum(p.numel() for p in model.parameters())
    tel.info(f"-> Toplam Parametre Sayısı: {param_count / 1e6:.2f} M")
    tel.record_metric("param_count_M", param_count / 1e6)

    # Accumulation hesabı
    accum = max(1, TARGET_EFFECTIVE_BATCH // micro_batch)
    effective_batch = micro_batch * accum
    tokens_per_step = effective_batch * CONTEXT_LEN

    tel.info(f"-> Micro-Batch / Accum     : {micro_batch} / {accum} (Efektif: {effective_batch})")
    tel.info(f"-> Adım Başına Token       : {tokens_per_step:,}")

    # Ortak Validation Veri Seti (Hızlı, 1500 blok)
    val_ds = ds_manager.get_validation_dataset(ready_data_dir, CONTEXT_LEN, max_val_samples=1500)

    # Dynamo hata toleransı
    if hasattr(torch, "_dynamo"):
        try:
            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass

    # ==========================================================================
    # 6. AŞAMALI (CURRICULUM) EĞİTİM DÖNGÜSÜ
    # ==========================================================================
    # Sırasıyla Stage 1 -> Stage 2 -> Stage 3 yürütülür.
    # Her aşama bittiğinde nihai ağırlıklar atomik kaydedilir ve bir sonraki aşamaya aktarılır.
    # Kesintiye uğrarsa StageResumeOrchestrator son adımdan devam ettirir.
    # ==========================================================================
    previous_stage_final_path: Optional[str] = None

    for stage_idx, stage_info in enumerate(STAGE_SPECS):
        stage_id = stage_info["id"]
        stage_name = stage_info["name"]
        stage_file = stage_info["file"]
        stage_lr = stage_info["learning_rate"]
        stage_wd = stage_info["weight_decay"]
        stage_max_steps = stage_info["max_steps"]
        stage_epochs = stage_info["epochs"]
        stage_warmup = stage_info["warmup_ratio"]

        tel.info("\n" + "=" * 80)
        tel.info(f"BAŞLATILIYOR: {stage_name}")
        tel.info(f"Hedef Token : {stage_info['tokens_approx']} | LR: {stage_lr} | Weight Decay: {stage_wd}")
        tel.info("=" * 80)

        # Stage Dizinleri
        stage_run_name = f"{RUN_NAME}_{stage_id}"
        LOCAL_STAGE_CKPT = os.path.join(LOCAL_BASE, f"ckpt_{stage_run_name}")
        DRIVE_STAGE_CKPT = os.path.join(DRIVE_BASE, f"ckpt_{stage_run_name}")
        LOCAL_STAGE_FINAL = os.path.join(LOCAL_BASE, f"final_{stage_run_name}")
        DRIVE_STAGE_FINAL = os.path.join(DRIVE_BASE, f"final_{stage_run_name}")
        LOCAL_STAGE_LOGS = os.path.join(LOCAL_BASE, f"logs_{stage_run_name}")
        DRIVE_STAGE_LOGS = os.path.join(DRIVE_BASE, "logs", stage_run_name)

        os.makedirs(LOCAL_STAGE_CKPT, exist_ok=True)
        os.makedirs(DRIVE_STAGE_CKPT, exist_ok=True)
        os.makedirs(LOCAL_STAGE_LOGS, exist_ok=True)
        os.makedirs(DRIVE_STAGE_LOGS, exist_ok=True)

        # 1. Bu stage daha önce tamamen tamamlanmış mı?
        stage_complete_marker = os.path.join(DRIVE_STAGE_FINAL, "_COMPLETE")
        if os.path.exists(stage_complete_marker):
            tel.info(f"-> [ATLANDI] {stage_id.upper()} zaten tamamlanmış görünüyor: {DRIVE_STAGE_FINAL}")
            previous_stage_final_path = DRIVE_STAGE_FINAL
            continue

        # 2. Önceki aşamadan model ağırlıklarını devral (veya sıfırdan başla)
        if previous_stage_final_path is not None:
            tel.info(f"-> Önceki aşamanın ({previous_stage_final_path}) ağırlıkları yükleniyor...")
            load_source = LOCAL_STAGE_FINAL if os.path.exists(LOCAL_STAGE_FINAL) else previous_stage_final_path
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model = LlamaForCausalLM.from_pretrained(
                load_source,
                attn_implementation=active_backend,
                **dtype_kwarg(torch.bfloat16 if specs["use_bf16"] else torch.float16)
            ).cuda()
            if USE_GRAD_CHECKPOINTING:
                model.gradient_checkpointing_enable()
            tel.info("-> Ağırlık aktarımı başarıyla tamamlandı.")

        # 3. Stage Veri Setini Yükle
        train_ds = ds_manager.get_stage_dataset(ready_data_dir, stage_file, CONTEXT_LEN)

        # 4. Resume Noktası Bul
        resume_orchestrator = StageResumeOrchestrator(
            stage_id=stage_id,
            local_dir=LOCAL_STAGE_CKPT,
            drive_dir=DRIVE_STAGE_CKPT,
            arch_sig=arch_sig,
            telemetry=tel
        )
        last_checkpoint = resume_orchestrator.find_resume_point()

        # 5. TrainingArguments Yapılandırması
        active_optim = OPTIMIZER_TYPE
        if active_optim == "adamw_torch_fused" and not specs["use_tf32"]:
            active_optim = "adamw_torch"

        ta_params = inspect.signature(TrainingArguments.__init__).parameters
        kwargs = {
            "output_dir": LOCAL_STAGE_CKPT,
            "overwrite_output_dir": False,
            "num_train_epochs": stage_epochs if stage_max_steps <= 0 else 100,
            "max_steps": stage_max_steps,
            "per_device_train_batch_size": micro_batch,
            "per_device_eval_batch_size": min(micro_batch, 8),
            "gradient_accumulation_steps": accum,
            "learning_rate": stage_lr,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": stage_warmup,
            "weight_decay": stage_wd,
            "max_grad_norm": 1.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1e-8,
            "optim": active_optim,
            "bf16": specs["use_bf16"],
            "fp16": specs["use_fp16"],
            "tf32": specs["use_tf32"],
            "gradient_checkpointing": USE_GRAD_CHECKPOINTING,
            "save_steps": SAVE_STEPS,
            "eval_steps": EVAL_STEPS,
            "save_total_limit": 5,
            "load_best_model_at_end": False,
            "prediction_loss_only": True,
            "dataloader_num_workers": min(4, os.cpu_count() or 2),
            "dataloader_pin_memory": True,
            "dataloader_persistent_workers": True,
            "dataloader_prefetch_factor": 4,
            "remove_unused_columns": False,
            "torch_compile": False,
            "logging_steps": LOGGING_STEPS,
            "logging_first_step": True,
            "logging_dir": LOCAL_STAGE_LOGS,
            "seed": 42 + stage_idx,
            "data_seed": 42 + stage_idx,
            "report_to": "none",
        }
        kwargs = {k: v for k, v in kwargs.items() if k in ta_params}
        if "eval_strategy" in ta_params:
            kwargs["eval_strategy"] = "steps"
        elif "evaluation_strategy" in ta_params:
            kwargs["evaluation_strategy"] = "steps"
        if "save_strategy" in ta_params:
            kwargs["save_strategy"] = "steps"

        training_args = TrainingArguments(**kwargs)

        # 6. Atomic Drive Publisher Bağla
        publisher = StageAtomicDrivePublisher(
            stage_id=stage_id,
            drive_ckpt_dir=DRIVE_STAGE_CKPT,
            local_logs_dir=LOCAL_STAGE_LOGS,
            drive_logs_dir=DRIVE_STAGE_LOGS,
            keep_last=2,
            telemetry=tel,
            drive_sync_steps=DRIVE_SYNC_STEPS,
        )

        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_ds,
            "eval_dataset": val_ds,
            "data_collator": collator,
            "callbacks": [publisher],
        }
        if "processing_class" in inspect.signature(Trainer.__init__).parameters:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = Trainer(**trainer_kwargs)

        # 7. Eğitimi Başlat
        tel.info(f"[{stage_id.upper()}] Eğitim başlıyor (Mod: {'RESUME' if last_checkpoint else 'FRESH'})...")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

        # 8. Stage Sonu Evaluation
        evaluator = ModelEvaluator(tel)
        eval_loss, eval_ppl = evaluator.evaluate(trainer, val_ds)

        # 9. Atomic Stage Final Model Save
        tel.info(f"[{stage_id.upper()}] AŞAMA NİHAİ MODEL KAYDI (ATOMIC)")
        if os.path.exists(LOCAL_STAGE_FINAL):
            shutil.rmtree(LOCAL_STAGE_FINAL)
        os.makedirs(LOCAL_STAGE_FINAL, exist_ok=True)

        raw_model = getattr(trainer.model, "_orig_mod", trainer.model)
        if hasattr(raw_model, "config"):
            raw_model.config.use_cache = True
        if hasattr(raw_model, "gradient_checkpointing_disable"):
            raw_model.gradient_checkpointing_disable()

        trainer.save_model(LOCAL_STAGE_FINAL)
        tokenizer.save_pretrained(LOCAL_STAGE_FINAL)

        stage_manifest = {
            "version": RUN_VERSION,
            "stage": stage_id,
            "stage_name": stage_name,
            "profile": PROFILE,
            "arch_sig": arch_sig,
            "metrics": {
                **getattr(train_result, "metrics", {}),
                "eval_loss": eval_loss,
                "perplexity": eval_ppl,
            },
            "completed_at": datetime.now().isoformat(),
        }
        with open(os.path.join(LOCAL_STAGE_FINAL, "stage_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(stage_manifest, f, indent=2, ensure_ascii=False, default=str)
        with open(os.path.join(LOCAL_STAGE_FINAL, "_COMPLETE"), "w") as f:
            f.write(f"stage={stage_id}\ncompleted=true\n")

        # Atomic Drive Aktarımı
        STAGE_FINAL_TMP = DRIVE_STAGE_FINAL + ".partial"
        if os.path.exists(STAGE_FINAL_TMP):
            shutil.rmtree(STAGE_FINAL_TMP, ignore_errors=True)
        if os.path.exists(DRIVE_STAGE_FINAL):
            shutil.rmtree(DRIVE_STAGE_FINAL, ignore_errors=True)
        shutil.copytree(LOCAL_STAGE_FINAL, STAGE_FINAL_TMP)
        try:
            os.replace(STAGE_FINAL_TMP, DRIVE_STAGE_FINAL)
        except OSError:
            if os.path.exists(DRIVE_STAGE_FINAL):
                shutil.rmtree(DRIVE_STAGE_FINAL, ignore_errors=True)
            shutil.copytree(STAGE_FINAL_TMP, DRIVE_STAGE_FINAL, dirs_exist_ok=True)
            shutil.rmtree(STAGE_FINAL_TMP, ignore_errors=True)

        tel.info(f"-> {stage_name} başarıyla tamamlandı ve Drive'a aktarıldı: {DRIVE_STAGE_FINAL}")
        previous_stage_final_path = DRIVE_STAGE_FINAL

        # Belleği temizle (bir sonraki stage için)
        del trainer, train_ds
        gc.collect()
        torch.cuda.empty_cache()

    # ==========================================================================
    # 7. TÜM AŞAMALAR TAMAMLANDI: NİHAİ BASE-V2 YAYINI & BENCHMARK
    # ==========================================================================
    FINAL_MASTER_DRIVE = os.path.join(DRIVE_BASE, f"final_{RUN_NAME}_master")
    tel.info("\n" + "=" * 80)
    tel.info("TÜM 3 AŞAMA BAŞARIYLA TAMAMLANDI! NİHAİ BASE-V2 HAZIRLANIYOR.")
    tel.info(f"Nihai Konum: {FINAL_MASTER_DRIVE}")
    tel.info("=" * 80)

    if previous_stage_final_path and os.path.exists(previous_stage_final_path):
        if os.path.exists(FINAL_MASTER_DRIVE):
            shutil.rmtree(FINAL_MASTER_DRIVE, ignore_errors=True)
        shutil.copytree(previous_stage_final_path, FINAL_MASTER_DRIVE)

    # Nihai log senkronu
    tel.save_manifest(os.path.join(FINAL_MASTER_DRIVE, "telemetry_manifest.json"))

    # Inference Benchmark Smoke Test
    dtype = torch.bfloat16 if specs["use_bf16"] else torch.float16
    evaluator = ModelEvaluator(tel)
    evaluator.inference_smoke_test(FINAL_MASTER_DRIVE, tokenizer, active_backend, dtype)

    tel.info("=" * 80)
    tel.info(f"TURKLM {RUN_VERSION} - TÜM OPERASYON BAŞARIYLA TAMAMLANDI!")
    tel.info(f"Model Drive Konumu: {FINAL_MASTER_DRIVE}")
    tel.info("=" * 80)


if __name__ == "__main__":
    main()
