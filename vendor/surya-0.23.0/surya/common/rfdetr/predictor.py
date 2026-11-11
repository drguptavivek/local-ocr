"""Vendored rf-detr (Roboflow RF-DETR) detection inference — no rfdetr package dependency.

Model definition copied (slimmed, detection-only) from the rfdetr package under
``surya/common/rfdetr/models`` + ``util``. This module is the thin runtime wrapper:
build the LWDETR architecture, load a fine-tuned checkpoint, and run predict() with the
same preprocessing/post-processing the rfdetr package uses (ImageNet-normalize, resize to
the model resolution, sigmoid top-k decode). Pure PyTorch — runs on cpu/mps/cuda.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from surya.common.rfdetr.models import PostProcess, build_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LARGE_ARGS = {   'amp': True,
    'aug_config': None,
    'aux_loss': True,
    'backbone_lora': False,
    'backbone_only': False,
    'batch_size': 2,
    'bbox_loss_coef': 5,
    'bbox_reparam': True,
    'ca_nheads': 16,
    'checkpoint_interval': 10,
    'clip_max_norm': 0.1,
    'cls_loss_coef': 1.0,
    'coco_path': None,
    'cutoff_epoch': 0,
    'dataset_dir': None,
    'dataset_file': 'coco',
    'dec_layers': 4,
    'dec_n_points': 2,
    'decoder_norm': 'LN',
    'dim_feedforward': 2048,
    'dist_url': 'env://',
    'do_benchmark': False,
    'do_random_resize_via_padding': False,
    'dont_save_weights': False,
    'drop_mode': 'standard',
    'drop_path': 0,
    'drop_schedule': 'constant',
    'dropout': 0,
    'early_stopping': True,
    'early_stopping_min_delta': 0.001,
    'early_stopping_patience': 10,
    'early_stopping_use_ema': False,
    'ema_decay': 0.9997,
    'ema_tau': 0,
    'encoder': 'dinov2_windowed_small',
    'encoder_only': False,
    'epochs': 12,
    'eval': False,
    'expanded_scales': False,
    'focal_alpha': 0.25,
    'force_no_pretrain': False,
    'fp16_eval': False,
    'freeze_batch_norm': False,
    'freeze_encoder': False,
    'giou_loss_coef': 2,
    'grad_accum_steps': 1,
    'gradient_checkpointing': False,
    'group_detr': 13,
    'hidden_dim': 256,
    'ia_bce_loss': True,
    'layer_norm': True,
    'license': 'Apache-2.0',
    'lite_refpoint_refine': True,
    'lr': 0.0001,
    'lr_component_decay': 1.0,
    'lr_drop': 11,
    'lr_encoder': 0.00015,
    'lr_min_factor': 0.0,
    'lr_scheduler': 'step',
    'lr_vit_layer_decay': 0.8,
    'mask_downsample_ratio': 4,
    'multi_scale': False,
    'num_classes': 90,
    'num_feature_levels': 1,
    'num_queries': 300,
    'num_select': 100,
    'num_windows': 2,
    'num_workers': 2,
    'out_feature_indexes': [3, 6, 9, 12],
    'output_dir': 'output',
    'patch_size': 16,
    'position_embedding': 'sine',
    'positional_encoding_size': 44,
    'pretrain_exclude_keys': None,
    'pretrain_keys_modify_to_load': None,
    'pretrained_distiller': None,
    'pretrained_encoder': None,
    'print_freq': 10,
    'projector_scale': ['P4'],
    'resolution': 704,
    'resume': '',
    'rms_norm': False,
    'sa_nheads': 8,
    'seed': 42,
    'segmentation_head': False,
    'set_cost_bbox': 5,
    'set_cost_class': 2,
    'set_cost_giou': 2,
    'square_resize_div_64': False,
    'start_epoch': 0,
    'sum_group_losses': False,
    'sync_bn': True,
    'two_stage': True,
    'use_cls_token': False,
    'use_ema': False,
    'use_position_supervised_loss': False,
    'use_varifocal_loss': False,
    'vit_encoder_num_layers': 12,
    'warmup_epochs': 1,
    'weight_decay': 0.0001,
    'window_block_indexes': None,
    'world_size': 1}


class RFDetrDetector:
    """Builds RF-DETR-Large and runs detection. ``predict`` returns, per image, a dict with
    ``boxes`` (xyxy pixels), ``scores``, ``labels`` (0-indexed class ids) as CPU tensors."""

    def __init__(self, weights_path: str, device: str = "cpu",
                 arch_args: Optional[dict] = None):
        args = SimpleNamespace(**{**LARGE_ARGS, **(arch_args or {})})
        args.device = device
        # Truthy so build_backbone sets load_dinov2_weights=False (no DINOv2 hub download);
        # the fine-tuned checkpoint is loaded manually below, not by build_model.
        args.pretrain_weights = weights_path
        self.device = torch.device(device)
        self.resolution = int(args.resolution)
        self.num_select = int(args.num_select)

        model = build_model(args)

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = ckpt["model"]
        # Match the checkpoint's class count (head was configured for the default 90).
        ckpt_num_classes = state["class_embed.bias"].shape[0]
        if ckpt_num_classes != args.num_classes + 1:
            model.reinitialize_detection_head(ckpt_num_classes)
        # Trim group-detr query params to the desired query count (no-op if already matching).
        num_desired = args.num_queries * args.group_detr
        for name in list(state.keys()):
            if name.endswith("refpoint_embed.weight") or name.endswith("query_feat.weight"):
                state[name] = state[name][:num_desired]
        model.load_state_dict(state, strict=False)

        self.model = model.eval().to(self.device)
        self.postprocess = PostProcess(num_select=self.num_select)

        # Optional capture of the encoder feature map (projector output, [B,C,F,F]) so the
        # reading-order head can cross-attend to it. Hook the LWDETR backbone projector.
        self._feat = {}
        for name, mod in self.model.named_modules():
            if name.endswith("projector"):
                mod.register_forward_hook(
                    lambda m, i, o: self._feat.__setitem__(
                        "f", (o[0] if isinstance(o, (tuple, list)) else o)))
                break

    @torch.inference_mode()
    def predict(self, images: List[Image.Image], threshold: float = 0.4,
                return_features: bool = False) -> List[dict]:
        if not images:
            return []
        tensors, sizes = [], []
        for img in images:
            img = img.convert("RGB")
            sizes.append((img.height, img.width))  # (h, w) for PostProcess scaling
            t = TF.to_tensor(img).to(self.device)
            t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
            t = TF.resize(t, (self.resolution, self.resolution))
            tensors.append(t)
        outputs = self.model(torch.stack(tensors, 0))
        feats = self._feat.get("f") if return_features else None  # [B,C,F,F]
        target_sizes = torch.tensor(sizes, device=self.device)
        results = self.postprocess(outputs, target_sizes=target_sizes)

        out = []
        for i, res in enumerate(results):
            keep = res["scores"] > threshold
            d = {
                "boxes": res["boxes"][keep].cpu(),
                "scores": res["scores"][keep].cpu(),
                "labels": res["labels"][keep].cpu(),
            }
            if return_features and feats is not None:
                d["features"] = feats[i].cpu()  # [C,F,F] for this page
            out.append(d)
        return out
