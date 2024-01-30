from gen import DEFAULT_PROMPT, PLACEHOLDER_TOKEN

def get_tokens(tokenizer, prompt: str = DEFAULT_PROMPT, placeholder_token: str = PLACEHOLDER_TOKEN):
    assert prompt.count(placeholder_token) == 1
    return tokenizer(DEFAULT_PROMPT, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.squeeze(0)

def get_uncond_tokens(tokenizer, prompt: str = ""):
    return tokenizer(prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.squeeze(0)
