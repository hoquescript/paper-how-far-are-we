import numpy as np
import torch


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_str(x):
    return "" if x is None else str(x)


DEFAULT_SEP_TOKEN = "</s>"


def resolve_separator_token(tokenizer) -> str:
    return tokenizer.sep_token or tokenizer.eos_token or DEFAULT_SEP_TOKEN
