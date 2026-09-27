"""
TURKLM 443M - RAG (Retrieval-Augmented Generation) & Gradio WebUI
Enhances the 443M parameter TURKLM model with live Turkish Wikipedia knowledge retrieval.
"""

import os
import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
import gradio as gr

# Try importing Wikipedia & SentenceTransformers
try:
    import wikipedia
    wikipedia.set_lang("tr")
    _WIKI_AVAILABLE = True
except ImportError:
    _WIKI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


class TurkishRAGPipeline:
    def __init__(self, model_path: str, tokenizer_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        print(f"[*] TURKLM modeli yukleniyor: {model_path} ({self.device})...")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        print("[+] Model ve Tokenizer basariyla yuklendi.")

    def retrieve_context(self, query: str, max_chars: int = 600) -> str:
        """Sorulan soruya gore Turkce Wikipedia'dan en alakali ozeti ceker."""
        if not _WIKI_AVAILABLE:
            return ""
        try:
            # En yakin basliklari ara
            search_results = wikipedia.search(query, results=2)
            if not search_results:
                return ""
            # Ilk sonucun ozetini al
            summary = wikipedia.summary(search_results[0], sentences=3)
            return summary[:max_chars].strip()
        except Exception as e:
            print(f"[RAG] Bilgi cekme atlandi: {e!r}")
            return ""

    def generate(self, user_query: str, use_rag: bool = True, max_new_tokens: int = 120, temperature: float = 0.3) -> str:
        context = ""
        if use_rag:
            context = self.retrieve_context(user_query)

        # TURKLM Chat & RAG Prompt Sablonu
        if context:
            prompt = (
                f"### Baglam Bilgisi:\n{context}\n\n"
                f"### Kullanici:\nYukaridaki baglam bilgisine dayanarak cevap ver: {user_query}\n\n"
                f"### Asistan:\n"
            )
        else:
            prompt = f"### Kullanici:\n{user_query}\n\n### Asistan:\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
                top_p=0.9,
                repetition_penalty=1.15,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Sadece asistanin urettigi yeni tokenlari decode et
        gen_tokens = output_tokens[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Ekstra baslik/prompt tekrarlarini temizle
        if "###" in response:
            response = response.split("###")[0].strip()

        return response, context


def launch_webui(model_path: str, tokenizer_path: str, port: int = 7860, share: bool = True):
    rag_pipe = TurkishRAGPipeline(model_path, tokenizer_path)

    def chat_fn(message, history, use_rag_box, temp_slider):
        answer, retrieved_ctx = rag_pipe.generate(message, use_rag=use_rag_box, temperature=temp_slider)
        if retrieved_ctx and use_rag_box:
            full_reply = f"**💡 Wikipedia RAG Baglami:**\n> *{retrieved_ctx}*\n\n**🤖 TURKLM Cevabi:**\n{answer}"
        else:
            full_reply = answer
        return full_reply

    with gr.Blocks(title="TURKLM 443M + RAG WebUI") as demo:
        gr.Markdown(
            "# 🇹🇷 TURKLM 443M: Turkish Language Model + RAG\n"
            "Sıfırdan eğitilmiş 443M parametreli Türkçe LLaMA modeli ve canlı Wikipedia RAG entegrasyonu."
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.ChatInterface(
                    fn=chat_fn,
                    additional_inputs=[
                        gr.Checkbox(label="RAG Aktif (Wikipedia'dan Canlı Bilgi Çek)", value=True),
                        gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.05, label="Temperature (Yaratıcılık)"),
                    ],
                )

    demo.launch(server_port=port, share=share)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TURKLM RAG WebUI")
    parser.add_argument("--model_path", type=str, default="/content/drive/MyDrive/Turkce_Tiny_LM/final_turklm_v10.2_sft_fix_a100_high_d401853a664e8a6b")
    parser.add_argument("--tokenizer_path", type=str, default="/content/drive/MyDrive/Turkce_Tiny_LM/tokenizer")
    parser.add_argument("--share", action="store_true", default=True, help="Colab disina acik link uret")
    args = parser.parse_args()

    launch_webui(args.model_path, args.tokenizer_path, share=args.share)
