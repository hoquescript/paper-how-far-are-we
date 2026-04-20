from .original import main as embeddding_model
from .k_fold import train as kfold_embedding_model
from .bagging import train as bagging_embedding_model

import torch

print(
    f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}"
)

__all__ = [embeddding_model, kfold_embedding_model, bagging_embedding_model]
