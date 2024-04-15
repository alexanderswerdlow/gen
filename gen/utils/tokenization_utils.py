from typing import Optional
from transformers import T5Tokenizer
from gen import DEFAULT_PROMPT, PLACEHOLDER_TOKEN


def _get_tokens(tokenizer, prompt, max_length: Optional[int] = None):
    if isinstance(tokenizer, T5Tokenizer):
        max_length = 24
    if max_length is None:
        max_length = tokenizer.model_max_length
    return tokenizer(prompt, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.squeeze(0)

def get_tokens(tokenizer, prompt: str = DEFAULT_PROMPT, placeholder_token: str = PLACEHOLDER_TOKEN):
    assert prompt.count(placeholder_token) == 1
    return _get_tokens(tokenizer, DEFAULT_PROMPT)

def get_uncond_tokens(tokenizer, prompt: str = ""):
    return _get_tokens(tokenizer, prompt)