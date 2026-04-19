from typing import List
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel


from .helper import resolve_separator_token


# -----------------------------
# CodeT5 embedding model wrapper
# -----------------------------
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
        """
        Extract normalized sequence embeddings from the CodeT5+ checkpoint.
        Returns array shape (n, embed_dim).
        """

        all_vecs = []
        n_batches = (len(texts) + batch_size - 1) // batch_size

        print(
            f"Embedding {len(texts)} samples on {self.device} "
            f"in {n_batches} batches (batch_size={batch_size})",
            flush=True,
        )

        for i in range(0, len(texts), batch_size):
            print(
                f"Embedding batch {i // batch_size + 1}/{n_batches}",
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
