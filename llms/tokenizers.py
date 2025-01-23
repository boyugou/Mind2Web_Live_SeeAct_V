import os
from typing import Any

import tiktoken
from transformers import LlamaTokenizer, AutoTokenizer  # type: ignore


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
            if 'internvl2' in model_name.lower() or "InternVL2" in model_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained("/disk2/models/internVL2-Llama3-76B/InternVL2-Llama3-76B")
            elif "llama3" in model_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained("/data/users/zhangjunlei/"
                                                               "download/models/output/"
                                                               "llama3.1_70b_lora_vwa_axt_512/ckpt_100/merged")
            else:
                self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        elif provider == "huggingface":
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            # turn off adding special tokens automatically
            self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
            self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
            self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        elif provider == "google":
            self.tokenizer = None  # Not used for input length computation, as Gemini is based on characters
        else:
            raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
