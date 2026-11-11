import os

# The rf-detr fast detector's DINOv2 pos-embed interpolation uses an antialiased-bicubic
# op that has no MPS kernel. PyTorch reads PYTORCH_ENABLE_MPS_FALLBACK at `import torch`,
# so set it here — before torch is imported — to let that one op fall back to CPU while
# the rest of the model runs on MPS. This is only effective when surya is imported before
# torch; when torch is already loaded, `_pick_device` (surya.common.rfdetr_torch) probes
# the op at runtime and falls back to CPU device selection instead. setdefault so we never
# clobber a value the user set deliberately.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
