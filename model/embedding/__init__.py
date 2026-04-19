from .original import main as embeddding_model
from .k_fold import train as kfold_embedding_model

__all__ = [embeddding_model, kfold_embedding_model]
