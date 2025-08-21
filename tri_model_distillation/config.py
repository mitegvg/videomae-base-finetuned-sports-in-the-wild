"""
Configuration classes for Tri-Model Asymmetric Distillation Framework
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from transformers import TrainingArguments
import logging

logger = logging.getLogger(__name__)


@dataclass
class TriModelConfig:
    """Configuration for the Tri-Model Asymmetric Distillation Framework"""
    
    # Classification type configuration
    classification_type: str = "binary"  # "binary", "multiclass", or "multilabel"
    num_labels: int = 2  # Number of target classes
    
    # Model configurations
    teacher_model_name: str = "MCG-NJU/videomae-base"
    assistant_model_path: Optional[str] = None  # Path to SSV2 pretrained model
    assistant_model_name: Optional[str] = None  # HuggingFace model name for assistant
    student_model_name: str = "MCG-NJU/videomae-base"
    
    # Student model configuration - PROGRESSIVE DISTILLATION ENABLED
    use_tiny_student: bool = True  # Create a tiny student model
    use_pretrained_student: bool = True
    pretrained_student_model: Optional[str] = None  # HF model name - will use student_model_name if None
    student_num_layers: int = 4  # Number of layers in student model (vs 12 in teacher)
    student_hidden_size: int = 384  # Hidden size for student (vs 768 in teacher)
    student_num_attention_heads: int = 6  # Attention heads (vs 12 in teacher)
    
    def __post_init__(self):
        """Post-initialization to set smart defaults"""
        # If pretrained_student_model is not set, default to student_model_name
        if self.pretrained_student_model is None:
            self.pretrained_student_model = self.student_model_name
    
    # Distillation weights - FIXED: Reduced feature weight to prevent overwhelming
    feature_distillation_weight: float = 0.1  # Drastically reduced from 1.0
    attention_distillation_weight: float = 0.1  # Reduced from 0.3 
    logits_distillation_weight: float = 0.8  # NEW: Logits distillation weight
    classification_loss_weight: float = 1.0  # Keep as primary focus
    
    # Temperature for distillation
    temperature: float = 6.0
    logits_temperature: float = 4.0  # NEW: Separate temperature for logits distillation
    
    # Feature alignment configurations
    align_hidden_states: bool = True
    align_attention_maps: bool = True
    hidden_layers_to_align: List[int] = field(default_factory=lambda: [-1, -2, -3])  # Last 3 layers
    
    # Assistant model configurations
    assistant_feature_weight: float = 0.8
    teacher_feature_weight: float = 1.0
    teacher_logits_weight: float = 1.0  # NEW: Weight for teacher logits
    assistant_logits_weight: float = 0.6  # NEW: Weight for assistant logits
    
    # Training configurations
    num_frames: int = 16
    image_size: int = 224
    mask_ratio: float = 0.0  # No masking for student during distillation
    
    # Dataset configurations
    dataset_root: str = "processed_dataset"
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"
    test_csv: str = "test.csv"
    
    # Output configurations
    output_dir: str = "tri_model_distilled_videomae"
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    
    # Logging
    logging_steps: int = 10
    save_total_limit: int = 3
    
    def validate_label_compatibility(self, label2id: dict) -> bool:
        """
        Validate that the configuration is compatible with the dataset labels.
        """
        if self.use_pretrained_student and self.pretrained_student_model:
            try:
                from transformers import VideoMAEConfig
                pretrained_config = VideoMAEConfig.from_pretrained(self.pretrained_student_model)
                
                if hasattr(pretrained_config, 'label2id') and pretrained_config.label2id:
                    pretrained_labels = set(pretrained_config.label2id.keys())
                    dataset_labels = set(label2id.keys())
                    
                    overlap = pretrained_labels.intersection(dataset_labels)
                    compatibility_ratio = len(overlap) / len(dataset_labels)
                    
                    if compatibility_ratio < 0.3:  # Less than 30% overlap
                        logger.warning(f"Low label compatibility: {compatibility_ratio:.1%}")
                        logger.warning("Consider using use_pretrained_student=False for better performance")
                        return False
                    else:
                        logger.info(f"Good label compatibility: {compatibility_ratio:.1%}")
                        return True
                        
            except Exception as e:
                logger.warning(f"Could not validate label compatibility: {e}")
                return False
        
        return True
    def get_classification_defaults(self) -> Dict[str, Any]:
        """Get default parameters based on classification type"""
        if self.classification_type == "binary":
            return {
                "temperature": 6.0,
                "logits_temperature": 3.0,  # NEW
                "feature_distillation_weight": 0.1,
                "attention_distillation_weight": 0.1,
                "logits_distillation_weight": 0.7,  # NEW
                "classification_loss_weight": 1.0,
                "assistant_feature_weight": 0.8,
                "assistant_logits_weight": 0.5,  # NEW
                "teacher_feature_weight": 1.0,
                "teacher_logits_weight": 1.0,  # NEW
            }
        elif self.classification_type == "multiclass":
            return {
                "temperature": 4.0,  # Lower for multiclass
                "logits_temperature": 4.0,  # NEW
                "feature_distillation_weight": 0.1,
                "attention_distillation_weight": 0.1,
                "logits_distillation_weight": 0.8,  # NEW - Higher for multiclass
                "classification_loss_weight": 1.0,
                "assistant_feature_weight": 0.6,  # Reduced for multiclass
                "assistant_logits_weight": 0.6,  # NEW
                "teacher_feature_weight": 1.0,
                "teacher_logits_weight": 1.0,  # NEW
            }
        elif self.classification_type == "multilabel":
            return {
                "temperature": 3.0,  # Even lower for multilabel
                "logits_temperature": 2.5,  # NEW - Lower for multilabel
                "feature_distillation_weight": 0.15,
                "attention_distillation_weight": 0.05,
                "logits_distillation_weight": 0.6,  # NEW
                "classification_loss_weight": 1.0,
                "assistant_feature_weight": 0.5,
                "assistant_logits_weight": 0.4,  # NEW
                "teacher_feature_weight": 1.0,
                "teacher_logits_weight": 1.0,  # NEW
            }
        else:
            raise ValueError(f"Unsupported classification_type: {self.classification_type}")
    
    def apply_classification_defaults(self):
        """Apply classification-specific defaults to the config"""
        defaults = self.get_classification_defaults()
        for key, value in defaults.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_training_args(self, **kwargs) -> TrainingArguments:
        """Convert to TrainingArguments for HuggingFace Trainer"""
        # Base/default arguments from config
        args = {
            "output_dir": self.output_dir,
            "save_strategy": self.save_strategy,
            "eval_strategy": self.eval_strategy,
            "logging_steps": self.logging_steps,
            "save_total_limit": self.save_total_limit,
            "remove_unused_columns": False,
            "dataloader_pin_memory": False,
        }

        # Normalize deprecated/alternate argument names for compatibility across
        # transformers versions (e.g., evaluation_strategy -> eval_strategy)
        if "evaluation_strategy" in kwargs and "eval_strategy" not in kwargs:
            args["eval_strategy"] = kwargs.pop("evaluation_strategy")

        # Merge the rest of kwargs, allowing explicit kwargs to override defaults
        args.update(kwargs)

        return TrainingArguments(**args)


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
