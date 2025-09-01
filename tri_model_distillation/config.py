"""
Configuration classes for Tri-Model Asymmetric Distillation Framework
"""

import os
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from transformers import TrainingArguments
import logging
import inspect

logger = logging.getLogger(__name__)


@dataclass
class TriModelConfig:
    """Configuration for the Tri-Model Asymmetric Distillation Framework"""
    
    # Classification type
    classification_type: str = "binary"
    num_labels: int = 2
    
    # Model names
    teacher_model_name: str = "MCG-NJU/videomae-base"
    assistant_model_path: Optional[str] = None
    assistant_model_name: Optional[str] = None
    student_model_name: str = "MCG-NJU/videomae-base"
    
    # Student shape
    use_tiny_student: bool = True
    use_pretrained_student: bool = True
    pretrained_student_model: Optional[str] = None
    student_num_layers: int = 4
    student_hidden_size: int = 384
    student_num_attention_heads: int = 6
    
    # Distillation weights
    feature_distillation_weight: float = 0.0
    attention_distillation_weight: float = 0.0
    logits_distillation_weight: float = 0.3
    classification_loss_weight: float = 1.0
    # Advanced / new distillation components
    temporal_delta_distillation_weight: float = 0.0  # Motion-aware (Delta) distillation
    temporal_delta_layers: List[int] = field(default_factory=lambda: [-1])
    enable_layer_fusion: bool = False  # Multi-layer fusion (ALP-KD style)
    layer_fusion_mode: str = "attention"  # 'attention' | 'uniform'
    layer_fusion_teacher_layers: List[int] = field(default_factory=list)  # empty => all teacher layers
    # New: assistant & multi-source fusion
    fusion_source: str = "teacher"  # 'teacher' | 'assistant' | 'both'
    layer_fusion_assistant_layers: List[int] = field(default_factory=list)  # empty => all assistant layers (when used)
    fusion_source_weighting: str = "learned"  # 'learned' | 'fixed'
    fusion_teacher_weight: float = 0.5  # used if weighting = fixed & fusion_source == 'both'
    fusion_assistant_weight: float = 0.5
    fusion_projection_dim: Optional[int] = None  # optional bottleneck dim for fusion projections
    attention_head_squeeze: bool = False  # SHD style head squeezing
    attention_head_squeeze_mode: str = "learned"  # 'learned' | 'avg'
    head_squeeze_ortho_weight: float = 1e-3  # orthogonality penalty weight for head mixing projection
    # Convenience explicit flags (optional overrides for ensuring outputs)
    temporal_delta_use_projection: bool = False  # (placeholder for future use)
    require_hidden_states: bool = False  # force output_hidden_states even if weights zero
    require_attentions: bool = False     # force output_attentions even if weights zero
    
    # Temperatures
    temperature: float = 6.0
    logits_temperature: float = 4.0
    
    # Alignment
    align_hidden_states: bool = True
    align_attention_maps: bool = True
    hidden_layers_to_align: List[int] = field(default_factory=lambda: [-1])
    
    # Source weights
    assistant_feature_weight: float = 0.8
    teacher_feature_weight: float = 1.0
    teacher_logits_weight: float = 1.0
    assistant_logits_weight: float = 0.6
    # NEW: explicit attention source weights (optional). If None -> derived from feature weights.
    teacher_attention_weight: Optional[float] = None
    assistant_attention_weight: Optional[float] = None
    # Logits KD gating
    kd_conf_threshold: float = 0.5          # assistant max prob threshold τ
    kd_min_keep_ratio: float = 0.25         # if fewer kept, relax threshold
    kd_dynamic_lower: bool = True           # allow automatic threshold relaxation
    
    # Data / video
    num_frames: int = 16
    image_size: int = 224
    mask_ratio: float = 0.0
    
    # Dataset
    dataset_root: str = "processed_dataset"
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"
    test_csv: str = "test.csv"
    
    # Output & scheduling
    output_dir: str = "tri_model_distilled_videomae"
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"      # canonical
    eval_strategy: Optional[str] = None     # deprecated alias
    
    # Logging
    logging_steps: int = 10
    save_total_limit: int = 3
    apply_defaults: bool = False
    
    def __post_init__(self):
        if self.eval_strategy and not self.evaluation_strategy:
            self.evaluation_strategy = self.eval_strategy
        if self.pretrained_student_model is None:
            self.pretrained_student_model = self.student_model_name
        # Only apply defaults if user explicitly asked
        if self.apply_defaults:
            self.apply_classification_defaults()
        # Derive attention weights if not explicitly provided (mirror feature weights, normalized)
        if self.teacher_attention_weight is None or self.assistant_attention_weight is None:
            t_fw = float(self.teacher_feature_weight)
            a_fw = float(self.assistant_feature_weight)
            denom = max(t_fw + a_fw, 1e-8)
            if self.teacher_attention_weight is None:
                self.teacher_attention_weight = t_fw / denom
            if self.assistant_attention_weight is None:
                self.assistant_attention_weight = a_fw / denom
        # Validation & info
        try:
            self.validate()
        except Exception:
            pass
    
    def get_classification_defaults(self) -> Dict[str, Any]:
        if self.classification_type == "binary":
            return {
                "temperature": 6.0,
                "logits_temperature": 3.0,
                "feature_distillation_weight": 0.0,
                "attention_distillation_weight": 0.0,
                "logits_distillation_weight": 0.2,
                "classification_loss_weight": 0.7,
                "assistant_feature_weight": 1.0,
                "assistant_logits_weight": 1.0,
                "teacher_feature_weight": 1.0,
                "teacher_logits_weight": 1.0,
            }
        elif self.classification_type == "multiclass":
            return {
                "temperature": 4.0,
                "logits_temperature": 4,
                "feature_distillation_weight": 0.03,
                "attention_distillation_weight": 0.2,
                "logits_distillation_weight": 0.5,
                "classification_loss_weight": 1.0,
                "assistant_feature_weight": 1.0,
                "assistant_logits_weight": 1.0,
                "teacher_feature_weight": 0.2,
                "teacher_logits_weight": 0.2,
            }
        elif self.classification_type == "multilabel":
            return {
                "temperature": 3.0,
                "logits_temperature": 2.5,
                "feature_distillation_weight": 0.0,
                "attention_distillation_weight": 0.0,
                "logits_distillation_weight": 0.2,
                "classification_loss_weight": 1.0,
                "assistant_feature_weight": 0.5,
                "assistant_logits_weight": 0.4,
                "teacher_feature_weight": 1.0,
                "teacher_logits_weight": 1.0,
            }
        else:
            raise ValueError(f"Unsupported classification_type: {self.classification_type}")
    
    def apply_classification_defaults(self):
        """Apply tuned defaults but do not override user-modified values.

        Heuristic: only overwrite if current value equals the dataclass field default.
        """
        defaults = self.get_classification_defaults()
        changed = {}
        for k, v in defaults.items():
            if hasattr(self, k):
                field_def = type(self).__dataclass_fields__[k].default
                current = getattr(self, k)
                if current == field_def:
                    setattr(self, k, v)
                    changed[k] = v
        if changed:
            logger.info("[TriModelConfig] Applied defaults: " + ", ".join(f"{k}={v}" for k,v in changed.items()))
    
    def to_training_args(self, **kwargs) -> TrainingArguments:
        base = {
            "output_dir": self.output_dir,
            "save_total_limit": self.save_total_limit,
            "remove_unused_columns": False,
            "dataloader_pin_memory": False,
        }
        # Prefer explicit overrides passed in call
        if "evaluation_strategy" in kwargs:
            eval_strat = kwargs["evaluation_strategy"]
        else:
            eval_strat = self.evaluation_strategy
        if "save_strategy" in kwargs:
            save_strat = kwargs["save_strategy"]
        else:
            save_strat = self.save_strategy
        
        # Compose
        base.update(kwargs)
        # Insert canonical keys (may be renamed below)
        base.setdefault("evaluation_strategy", eval_strat)
        base.setdefault("save_strategy", save_strat)
        
        # Introspect TrainingArguments to adapt
        sig = inspect.signature(TrainingArguments.__init__)
        params = sig.parameters
        
        if "evaluation_strategy" not in params and "eval_strategy" in params:
            # Legacy variant
            base["eval_strategy"] = base.pop("evaluation_strategy")
        # Drop unsupported keys gracefully
        for maybe in ("evaluation_strategy", "save_strategy", "logging_strategy"):
            if maybe in base and maybe not in params:
                base.pop(maybe)
        
        # Optional debug
        if "eval_strategy" in base and "evaluation_strategy" not in params:
            logger.debug("[TriModelConfig] Using legacy eval_strategy fallback.")
        
        return TrainingArguments(**base)
    
    def active_require_hidden(self) -> bool:
        fusion_active = False
        if self.enable_layer_fusion:
            if self.fusion_source == 'teacher':
                fusion_active = True
            elif self.fusion_source == 'assistant':
                fusion_active = True
            elif self.fusion_source == 'both':
                fusion_active = True
        return bool(
            (self.feature_distillation_weight > 0)
            or (self.attention_distillation_weight > 0)
            or (self.temporal_delta_distillation_weight > 0)
            or fusion_active
        )
    
    def active_require_attn(self) -> bool:
        return (self.attention_distillation_weight > 0) or self.align_attention_maps
    
    def validate(self):
        errors = []
        valid = set(self.__dataclass_fields__.keys())
        unexpected = [k for k in self.__dict__.keys() if k not in valid]
        # Fusion validation (only if enabled)
        if self.enable_layer_fusion:
            if self.fusion_source not in {"teacher", "assistant", "both"}:
                errors.append(
                    f"fusion_source must be 'teacher','assistant','both'; got {self.fusion_source}"
                )
            if self.fusion_source == 'teacher':
                if self.layer_fusion_teacher_layers and any(l < 0 for l in self.layer_fusion_teacher_layers):
                    errors.append("layer_fusion_teacher_layers must be non-negative indices")
            if self.fusion_source == 'assistant':
                if self.layer_fusion_assistant_layers and any(l < 0 for l in self.layer_fusion_assistant_layers):
                    errors.append("layer_fusion_assistant_layers must be non-negative indices")
            if self.fusion_source == 'both':
                if self.fusion_source_weighting not in {"learned", "fixed"}:
                    errors.append(
                        f"fusion_source_weighting must be 'learned' or 'fixed'; got {self.fusion_source_weighting}"
                    )
                if self.fusion_source_weighting == 'fixed':
                    total = self.fusion_teacher_weight + self.fusion_assistant_weight
                    if not math.isclose(total, 1.0, rel_tol=1e-3):
                        errors.append(
                            f"For fixed weighting fusion_teacher_weight + fusion_assistant_weight must sum to 1 (got {total:.4f})"
                        )
        if unexpected:
            import warnings
            warnings.warn(f"TriModelConfig: Ignoring unexpected fields: {unexpected}")
        if errors:
            raise ValueError("TriModelConfig validation errors: \n" + "\n".join(errors))
        logger.info(
            f"[TriModelConfig] Active components => "
            f"logits:{self.logits_distillation_weight>0} "
            f"features:{self.feature_distillation_weight>0} "
            f"attn:{self.attention_distillation_weight>0} "
            f"attn_w=({self.teacher_attention_weight:.3f},{self.assistant_attention_weight:.3f}) "
            f"apply_defaults={self.apply_defaults}"
        )


@dataclass
class SSV2ModelConfig:
    """Configuration for downloading and using SSV2 pretrained models"""
    
    # AMD MODEL_ZOO SSV2 model URLs
    ssv2_vit_s_pretrain_url: str = "https://drive.google.com/file/d/12hEVVdrFv0VNAIqAIZ0UTeicn-tm4jva/view?usp=drive_link"
    ssv2_vit_s_finetune_url: str = "https://drive.google.com/file/d/1ynDyu3K_INoZjaNLzFIaYCo6kjBOwG__/view?usp=sharing"
    ssv2_vit_b_pretrain_url: str = "https://drive.google.com/file/d/13EOb--vymBQpLNbztcN7wkdxXJiVfyXa/view?usp=sharing"
    ssv2_vit_b_finetune_url: str = "https://drive.google.com/file/d/1zc3ITIp4rR-dSelH0KaqMA9ahyju7sV2/view?usp=drive_link"
    
    # Local storage paths
    models_dir: str = "ssv2_models"
    default_backbone: str = "ViT-B"  # ViT-S or ViT-B
    use_finetuned: bool = True  # Use fine-tuned version instead of pre-trained
    num_classes: int = 174  # SSV2 has 174 classes
    
    def get_model_path(self, backbone: str = None, use_finetuned: bool = None) -> str:
        """Get local path for the specified model"""
        backbone = backbone or self.default_backbone
        use_finetuned = use_finetuned if use_finetuned is not None else self.use_finetuned
        
        model_type = "finetune" if use_finetuned else "pretrain"
        model_name = f"ssv2_vit_{backbone.lower().replace('-', '_')}_{model_type}.pth"
        
        return os.path.join(self.models_dir, model_name)
    
    def get_download_url(self, backbone: str = None, use_finetuned: bool = None) -> str:
        """Get download URL for the specified model"""
        backbone = backbone or self.default_backbone
        use_finetuned = use_finetuned if use_finetuned is not None else self.use_finetuned
        
        if backbone == "ViT-S":
            return self.ssv2_vit_s_finetune_url if use_finetuned else self.ssv2_vit_s_pretrain_url
        elif backbone == "ViT-B":
            return self.ssv2_vit_b_finetune_url if use_finetuned else self.ssv2_vit_b_pretrain_url
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'ViT-S' or 'ViT-B'")
