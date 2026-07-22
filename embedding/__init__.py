import numpy as np

from dataclasses import dataclass
from datetime import datetime
from typing import List

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_str(x):
    return "" if x is None else str(x)


DEFAULT_SEP_TOKEN = "</s>"


def resolve_separator_token(tokenizer) -> str:
    return tokenizer.sep_token or tokenizer.eos_token or DEFAULT_SEP_TOKEN


@dataclass
class CodeEmbedder:
    model_name: str = "Salesforce/codet5p-110m-embedding"
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.separator_token = resolve_separator_token(self.tokenizer)
        # Newer transformers T5Stack expects config.is_decoder; the hub's
        # CodeT5pEmbeddingConfig does not define it (encoder-only checkpoint).
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        if not hasattr(config, "is_decoder"):
            config.is_decoder = False
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True, config=config
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Extract normalized sequence embeddings. Returns array shape (n, embed_dim)."""
        all_vecs = []
        n_batches = (len(texts) + batch_size - 1) // batch_size

        print(
            f"[{ts()}] Embedding {len(texts)} samples on {self.device} "
            f"in {n_batches} batches (batch_size={batch_size})",
            flush=True,
        )

        for i in range(0, len(texts), batch_size):
            if i % 100 == 0:
                print(
                    f"[{ts()}] Embedding batch {i // batch_size + 1}/{n_batches}",
                    flush=True,
                )
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            embeddings = self.model(**enc)
            all_vecs.append(embeddings.detach().cpu().numpy())

        return np.vstack(all_vecs)
