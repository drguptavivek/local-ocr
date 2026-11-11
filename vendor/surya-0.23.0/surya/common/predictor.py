from typing import Optional

import torch

from surya.common.load import ModelLoader
from surya.settings import settings


class BasePredictor:
    model_loader_cls = ModelLoader
    batch_size: Optional[int] = None
    default_batch_sizes = {"cpu": 1, "mps": 1, "cuda": 1}
    torch_dtype = settings.MODEL_DTYPE

    @property
    def disable_tqdm(self) -> bool:
        return self._disable_tqdm

    @disable_tqdm.setter
    def disable_tqdm(self, value: bool) -> None:
        self._disable_tqdm = bool(value)

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: torch.device | str | None = settings.TORCH_DEVICE_MODEL,
        dtype: Optional[torch.dtype | str] = None,
        attention_implementation: Optional[str] = None,
    ):
        if dtype is None:
            dtype = self.torch_dtype

        loader = self.model_loader_cls(checkpoint)
        self.model = loader.model(device, dtype, attention_implementation)
        self.processor = loader.processor()
        self._disable_tqdm = settings.DISABLE_TQDM

    def to(self, device_dtype: torch.device | str | None = None):
        if hasattr(self, "model") and self.model:
            self.model.to(device_dtype)
            return
        # Predictors that don't own a torch model (e.g. VLM-backed predictors that
        # rely on an external server) treat .to() as a no-op.
        if hasattr(self, "manager") and self.manager is not None:
            return
        raise ValueError("Model not loaded")

    def get_batch_size(self):
        batch_size = self.batch_size
        if batch_size is None:
            batch_size = self.default_batch_sizes["cpu"]
            if settings.TORCH_DEVICE_MODEL in self.default_batch_sizes:
                batch_size = self.default_batch_sizes[settings.TORCH_DEVICE_MODEL]
        return batch_size

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
