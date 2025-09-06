# src/model.py
import os
import logging
from pathlib import Path
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

# Try import BitsAndBytesConfig (requires bitsandbytes & recent transformers)
try:
    from transformers import BitsAndBytesConfig
    _BNB_AVAILABLE = True
except Exception:
    BitsAndBytesConfig = None
    _BNB_AVAILABLE = False

# Optional PEFT/LoRA support
try:
    from peft import get_peft_model, LoraConfig, TaskType
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False

import config


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class TransformerModel:
    """
    Causal LM wrapper with optional:
      - 4-bit quantization via bitsandbytes
      - LoRA/PEFT adapter wrapping
    """

    def __init__(self, model_name: str | None = None, max_length: int | None = None,
                 use_peft: bool = False, use_auth_token: bool | str = True):
        self.model_name = model_name or config.MODEL_NAME
        self.max_length = max_length or config.MAX_LENGTH
        self.device = get_device()
        self.use_peft_flag = use_peft and _PEFT_AVAILABLE and config.USE_PEFT
        self.auth = use_auth_token  # bool or token string

        logging.info(f"[TransformerModel] Loading tokenizer for {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            token=self.auth  # new token argument
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        # Attempt to load model with 4-bit quantization if bitsandbytes available.
        self.model = None
        if _BNB_AVAILABLE:
            logging.info("[TransformerModel] Attempting 4-bit quantized load (bitsandbytes).")
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    quantization_config=bnb_config,
                    token=self.auth,
                )
                logging.info("[TransformerModel] 4-bit model loaded with device_map='auto'.")
            except Exception as e:
                logging.warning(f"[TransformerModel] 4-bit load failed ({e}). Falling back to standard load.")
                self.model = None

        # Fallback standard load
        if self.model is None:
            logging.info("[TransformerModel] Loading model (standard precision / device_map auto).")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                    token=self.auth,
                    low_cpu_mem_usage=True
                )
            except Exception as e:
                logging.error(f"[TransformerModel] Failed to load model: {e}")
                raise

        # resize embeddings if tokenizer changed (pad token added)
        try:
            self.model.resize_token_embeddings(len(self.tokenizer))
        except Exception:
            # some quantized wrappers may not support resize; ignore
            pass

        # Optionally wrap with PEFT/LoRA
        if self.use_peft_flag:
            logging.info("[TransformerModel] Configuring LoRA/PEFT adapters.")
            target_modules = ["q_proj", "v_proj", "o_proj"] if "llama" in self.model_name.lower() else ["q_proj", "v_proj", "o_proj"]
            peft_config = LoraConfig(
                r=config.LORA_R,
                lora_alpha=config.LORA_ALPHA,
                target_modules=target_modules,
                lora_dropout=config.LORA_DROPOUT,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            try:
                self.model = get_peft_model(self.model, peft_config)
                logging.info("[TransformerModel] LoRA applied.")
            except Exception as e:
                logging.warning(f"[TransformerModel] LoRA/PEFT wrapping failed: {e}")
                self.use_peft_flag = False

    def _model_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device(self.device)

    def generate_code(self, prompts, batch_size: int = 1,
                      max_new_tokens: int | None = None,
                      temperature: float | None = None,
                      top_p: float | None = None,
                      top_k: int | None = None,
                      num_beams: int | None = None):
        max_new_tokens = config.GEN_MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens
        temperature = config.GEN_TEMPERATURE if temperature is None else temperature
        top_p = config.GEN_TOP_P if top_p is None else top_p
        top_k = config.GEN_TOP_K if top_k is None else top_k
        num_beams = config.GEN_NUM_BEAMS if num_beams is None else num_beams

        self.model.eval()
        outputs = []

        model_device = self._model_device()

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            enc = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            enc = {k: v.to(model_device) for k, v in enc.items()}

            with torch.no_grad():
                gen_ids = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=(temperature > 0.0),
                )

            for j, g in enumerate(gen_ids):
                text = self.tokenizer.decode(g, skip_special_tokens=True)
                prompt_text = batch_prompts[j].strip()
                generated = text[len(prompt_text):].strip() if text.startswith(prompt_text) else text.strip()
                outputs.append(generated)

        return outputs

    def save(self, out_dir: str):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"[TransformerModel] Saving tokenizer & model to {out_dir}")
        self.tokenizer.save_pretrained(str(out_dir))
        try:
            self.model.save_pretrained(str(out_dir))
        except Exception as e:
            logging.warning(f"[TransformerModel] model.save_pretrained failed: {e}. Using torch.save fallback.")
            try:
                torch.save(self.model.state_dict(), str(out_dir / "pytorch_model.bin"))
            except Exception as e2:
                logging.error(f"[TransformerModel] Fallback save failed: {e2}")

    @classmethod
    def load_from(cls, out_dir: str, use_peft: bool = False, use_auth_token: bool | str = True):
        return cls(model_name=out_dir, max_length=config.MAX_LENGTH, use_peft=use_peft, use_auth_token=use_auth_token)
