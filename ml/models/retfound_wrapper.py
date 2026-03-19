import torch
import torch.nn as nn
from pathlib import Path
from peft import LoraConfig, get_peft_model, TaskType

import sys
sys.path.insert(0, str(Path.home() / "netra/models/RETFound"))
import models_vit
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_


class RETFoundDRGrader(nn.Module):
    def __init__(self, num_classes=5, pretrained_path=None, model_variant="mae", lora_rank=16, lora_alpha=32, lora_dropout=0.1, use_lora=True, drop_path=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.model_variant = model_variant

        if model_variant == "mae":
            self.backbone = models_vit.__dict__["vit_large_patch16"](num_classes=num_classes, drop_path_rate=drop_path, global_pool=True)
        elif model_variant == "dinov2":
            self.backbone = models_vit.__dict__["retfound_dinov2"](num_classes=num_classes, drop_path_rate=drop_path, global_pool=True)

        if pretrained_path:
            self._load_pretrained(pretrained_path)
        if use_lora:
            self._apply_lora(lora_rank, lora_alpha, lora_dropout)

    def _load_pretrained(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_model = checkpoint.get("model", checkpoint)
        state_dict = self.backbone.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict.get(k, torch.tensor([])).shape:
                del checkpoint_model[k]
        interpolate_pos_embed(self.backbone, checkpoint_model)
        msg = self.backbone.load_state_dict(checkpoint_model, strict=False)
        print(f"Loaded RETFound weights. Missing keys: {msg.missing_keys}")
        trunc_normal_(self.backbone.head.weight, std=2e-5)

    def _apply_lora(self, rank, alpha, dropout):
        lora_config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"], lora_dropout=dropout, bias="none", modules_to_save=["head", "fc_norm"])
        self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone.print_trainable_parameters()

    def forward(self, x):
        return self.backbone(x)

    def get_features(self, x):
        features = self.backbone.base_model.model.forward_features(x)
        return features

    def merge_lora_and_save(self, save_path):
        merged = self.backbone.merge_and_unload()
        torch.save(merged.state_dict(), save_path)
        print(f"Saved merged model to {save_path}")

    @classmethod
    def from_finetuned(cls, checkpoint_path, device="mps"):
        model = cls(use_lora=False)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.backbone.load_state_dict(state_dict)
        model.eval()
        return model.to(device)
