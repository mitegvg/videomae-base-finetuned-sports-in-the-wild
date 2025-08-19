"""
Tri-Model Asymmetric Distillation Framework Models

This module implements the core models and framework for tri-model distillation,
including model loading, feature extraction, and integration with HuggingFace transformers.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import (
    VideoMAEForVideoClassification, 
    VideoMAEImageProcessor,
    VideoMAEConfig
)
import gdown
import logging

# Handle both relative and absolute imports
try:
    from .config import TriModelConfig, SSV2ModelConfig
except ImportError:
    from config import TriModelConfig, SSV2ModelConfig

logger = logging.getLogger(__name__)


class SSV2ModelLoader:
    """Utility class for downloading and loading SSV2 pretrained models from AMD MODEL_ZOO."""
    
    def __init__(self, config: SSV2ModelConfig):
        self.config = config
        os.makedirs(config.models_dir, exist_ok=True)
    
    def download_model(self, backbone: str = None, use_finetuned: bool = None) -> str:
        """
        Download SSV2 model from Google Drive if not already cached.
        
        Args:
            backbone: Model backbone ('ViT-S' or 'ViT-B')
            use_finetuned: Whether to use fine-tuned or pre-trained model
            
        Returns:
            Local path to the downloaded model
        """
        model_path = self.config.get_model_path(backbone, use_finetuned)
        
        if os.path.exists(model_path):
            logger.info(f"Model already exists at {model_path}")
            return model_path
        
        download_url = self.config.get_download_url(backbone, use_finetuned)
        
        # Extract file ID from Google Drive URL
        file_id = self._extract_file_id_from_url(download_url)
        
        logger.info(f"Downloading SSV2 model from {download_url}")
        try:
            gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
            logger.info(f"Successfully downloaded model to {model_path}")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            # Create a placeholder file to avoid repeated download attempts
            torch.save({}, model_path)
            logger.warning(f"Created placeholder file at {model_path}")
        
        return model_path
    
    def _extract_file_id_from_url(self, url: str) -> str:
        """Extract Google Drive file ID from sharing URL."""
        if "/file/d/" in url:
            return url.split("/file/d/")[1].split("/")[0]
        elif "id=" in url:
            return url.split("id=")[1].split("&")[0]
        else:
            raise ValueError(f"Cannot extract file ID from URL: {url}")
    
    def load_model(
        self, 
        backbone: str = None, 
        use_finetuned: bool = None,
        num_labels: int = 174  # SSV2 has 174 classes
    ) -> VideoMAEForVideoClassification:
        """
        Load SSV2 model with proper configuration.
        
        Args:
            backbone: Model backbone ('ViT-S' or 'ViT-B')
            use_finetuned: Whether to use fine-tuned or pre-trained model
            num_labels: Number of output classes
            
        Returns:
            Loaded VideoMAE model
        """
        model_path = self.download_model(backbone, use_finetuned)
        
        # Create model configuration based on backbone
        if backbone == "ViT-S":
            config = VideoMAEConfig(
                num_hidden_layers=12,
                hidden_size=384,
                intermediate_size=1536,
                num_attention_heads=6,
                num_labels=num_labels,
                image_size=224,
                num_frames=16
            )
        else:  # ViT-B
            config = VideoMAEConfig(
                num_hidden_layers=12,
                hidden_size=768,
                intermediate_size=3072,
                num_attention_heads=12,
                num_labels=num_labels,
                image_size=224,
                num_frames=16
            )
        
        # Initialize model
        model = VideoMAEForVideoClassification(config)
        
        # Load weights if available
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Handle potential key mismatches
            model_state_dict = model.state_dict()
            matched_state_dict = {}
            
            for key, value in state_dict.items():
                if key in model_state_dict and value.shape == model_state_dict[key].shape:
                    matched_state_dict[key] = value
                else:
                    logger.warning(f"Skipping key {key} due to shape mismatch or missing key")
            
            model.load_state_dict(matched_state_dict, strict=False)
            logger.info(f"Loaded SSV2 model weights from {model_path}")
            
        except Exception as e:
            logger.warning(f"Could not load weights from {model_path}: {e}")
            logger.info("Using randomly initialized model")
        
        return model


class TriModelDistillationFramework(nn.Module):
    """
    Main framework class that orchestrates the tri-model distillation process.
    
    Manages:
    1. Teacher model (VideoMAE base, frozen)
    2. Assistant model (SSV2 pretrained, frozen)  
    3. Student model (target for fine-tuning)
    """
    
    def __init__(
        self,
        config: TriModelConfig,
        num_labels: int,
        label2id: Dict[str, int],
        id2label: Dict[int, str]
    ):
        super().__init__()
        
        self.config = config
        self.num_labels = num_labels
        self.label2id = label2id
        self.id2label = id2label
        
        # Apply classification-specific defaults if not already set
        if hasattr(config, 'apply_classification_defaults'):
            config.apply_classification_defaults()
        
        logger.info(f"Initializing Tri-Model Distillation Framework for {config.classification_type} classification...")
        
        # Initialize models
        self._initialize_models()
        
        # Freeze teacher and assistant models
        self._freeze_model(self.teacher_model)
        if hasattr(self, 'assistant_model'):
            self._freeze_model(self.assistant_model)
    
    def _initialize_models(self):
        """Initialize teacher, assistant, and student models."""
        
        # 1. Initialize Teacher Model (VideoMAE base)
        logger.info("Loading teacher model...")
        self.teacher_model = VideoMAEForVideoClassification.from_pretrained(
            self.config.teacher_model_name,
            num_labels=self.num_labels,
            label2id=self.label2id,
            id2label=self.id2label,
            ignore_mismatched_sizes=True
        )
        self.teacher_model.config.mask_ratio = 0.0
        
        # 2. Initialize Assistant Model (SSV2 or HuggingFace)
        logger.info("Loading assistant model...")
        if self.config.assistant_model_name:
            # Load assistant model from HuggingFace
            try:
                logger.info(f"Loading assistant model from HuggingFace: {self.config.assistant_model_name}")
                self.assistant_model = VideoMAEForVideoClassification.from_pretrained(
                    self.config.assistant_model_name,
                    num_labels=self.num_labels,
                    label2id=self.label2id,
                    id2label=self.id2label,
                    ignore_mismatched_sizes=True
                )
            except Exception as e:
                logger.warning(f"Could not load assistant model from HuggingFace: {e}")
                logger.info("Using teacher model as assistant (fallback)")
                self.assistant_model = self.teacher_model
        elif self.config.assistant_model_path:
            # Load assistant model from local path (SSV2)
            ssv2_config = SSV2ModelConfig()
            ssv2_loader = SSV2ModelLoader(ssv2_config)
            
            try:
                self.assistant_model = ssv2_loader.load_model(
                    backbone="ViT-B", 
                    use_finetuned=True,
                    num_labels=174  # SSV2 original classes
                )
                
                # Adapt assistant model to target domain
                self._adapt_assistant_model()
                
            except Exception as e:
                logger.warning(f"Could not load assistant model from local path: {e}")
                logger.info("Using teacher model as assistant (fallback)")
                self.assistant_model = VideoMAEForVideoClassification.from_pretrained(
                    self.config.teacher_model_name,
                    num_labels=self.num_labels,
                    label2id=self.label2id,
                    id2label=self.id2label,
                    ignore_mismatched_sizes=True
                )
        else:
            logger.info("No assistant model specified, using teacher as assistant")
            self.assistant_model = self.teacher_model
        
        # 3. Initialize Student Model with FIXED label alignment
        logger.info("Loading student model with label alignment...")
        
        # CRITICAL FIX: Check for actual label mapping compatibility, not just count
        if self.config.use_pretrained_student and self.config.pretrained_student_model:
            try:
                # Get pretrained model's actual label mappings
                from transformers import VideoMAEConfig
                pretrained_config = VideoMAEConfig.from_pretrained(self.config.pretrained_student_model)
                pretrained_label2id = getattr(pretrained_config, 'label2id', None)
                
                # Check if the label mappings actually match
                if (pretrained_label2id and 
                    self._labels_actually_match(pretrained_label2id, self.label2id)):
                    logger.info("Label mappings actually match - loading pretrained model directly")
                    self.student_model = VideoMAEForVideoClassification.from_pretrained(
                        self.config.pretrained_student_model,
                        num_labels=self.num_labels,
                        label2id=self.label2id,
                        id2label=self.id2label,
                        ignore_mismatched_sizes=False
                    )
                else:
                    logger.info("❌ Label mappings don't match - applying label alignment")
                    self.student_model = self._load_student_with_label_alignment(
                        pretrained_label2id
                    )
                    
            except Exception as e:
                logger.error(f"Failed to load pretrained student with alignment: {e}")
                logger.info("Falling back to creating tiny student from scratch")
                self.student_model = self._create_tiny_student()
        else:
            # Create tiny student or load without pretrained weights
            self.student_model = load_student_model(
                config=self.config, 
                label2id=self.label2id, 
                id2label=self.id2label
            )
        
        # Configure student model
        self.student_model.config.mask_ratio = self.config.mask_ratio
        
        # Initialize image processor
        self.image_processor = VideoMAEImageProcessor.from_pretrained(
            self.config.teacher_model_name
        )
        
        # CRITICAL FIX: Ensure student model config has correct label mappings
        # This prevents the saved model from having generic LABEL_X labels
        self.student_model.config.label2id = self.label2id
        self.student_model.config.id2label = self.id2label
        self.student_model.config.num_labels = self.num_labels

    def _labels_actually_match(self, pretrained_label2id: dict, target_label2id: dict) -> bool:
        """Check if label mappings actually match (not just counts)."""
        if not pretrained_label2id or not target_label2id:
            return False
            
        # Check if all target labels exist in pretrained with same IDs
        for label, target_id in target_label2id.items():
            if label not in pretrained_label2id:
                logger.info(f"Label '{label}' not found in pretrained model")
                return False
            if pretrained_label2id[label] != target_id:
                logger.info(f"Label '{label}': pretrained ID {pretrained_label2id[label]} != target ID {target_id}")
                return False
        
        logger.info("All label mappings match perfectly!")
        return True

    def _load_student_with_label_alignment(self, pretrained_label2id: dict):
        """Load student model and align labels properly."""
        logger.info("🔧 Applying label alignment to pretrained student model...")
        
        # Load the pretrained model with original configuration
        model = VideoMAEForVideoClassification.from_pretrained(self.config.pretrained_student_model)
        
        # Check which labels can be aligned
        target_labels = set(self.label2id.keys())
        pretrained_labels = set(pretrained_label2id.keys())
        aligned_labels = target_labels.intersection(pretrained_labels)
        
        logger.info(f"Can align {len(aligned_labels)}/{len(target_labels)} labels: {sorted(aligned_labels)}")
        logger.info(f"Missing labels: {sorted(target_labels - pretrained_labels)}")
        
        if len(aligned_labels) == 0:
            logger.warning("No labels can be aligned - creating new classifier head")
            # Replace entire classifier
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, self.num_labels)
            nn.init.normal_(model.classifier.weight, std=0.02)
            nn.init.zeros_(model.classifier.bias)
        else:
            # Create new classifier and copy aligned weights
            old_classifier = model.classifier
            new_classifier = nn.Linear(old_classifier.in_features, self.num_labels)
            
            # Initialize new classifier
            nn.init.normal_(new_classifier.weight, std=0.02)
            nn.init.zeros_(new_classifier.bias)
            
            # Copy weights for aligned labels
            with torch.no_grad():
                aligned_count = 0
                for target_label, target_idx in self.label2id.items():
                    if target_label in pretrained_label2id:
                        pretrained_idx = pretrained_label2id[target_label]
                        new_classifier.weight[target_idx] = old_classifier.weight[pretrained_idx]
                        new_classifier.bias[target_idx] = old_classifier.bias[pretrained_idx]
                        aligned_count += 1
                        logger.debug(f"Aligned '{target_label}': {pretrained_idx} -> {target_idx}")
            
            model.classifier = new_classifier
            logger.info(f"✅ Successfully aligned {aligned_count} labels from pretrained model")
        
        # Update model config
        model.config.num_labels = self.num_labels
        model.config.label2id = self.label2id
        model.config.id2label = self.id2label
        
        return model

    def _create_tiny_student(self):
        """Create a tiny student model from scratch."""
        from transformers import VideoMAEConfig
        
        config = VideoMAEConfig(
            num_hidden_layers=4,
            hidden_size=384,
            num_attention_heads=6,
            num_labels=self.num_labels,
            label2id=self.label2id,
            id2label=self.id2label
        )
        
        model = VideoMAEForVideoClassification(config)
        model.apply(model._init_weights)
        
        logger.info("Created tiny student model from scratch")
        return model
    
    def _adapt_assistant_model(self):
        """Adapt assistant model's classifier to target domain."""
        if hasattr(self.assistant_model, 'classifier'):
            # Replace classifier head to match target number of classes
            in_features = self.assistant_model.classifier.in_features
            self.assistant_model.classifier = nn.Linear(in_features, self.num_labels)
            
            # Initialize with small random weights
            nn.init.normal_(self.assistant_model.classifier.weight, std=0.02)
            nn.init.zeros_(self.assistant_model.classifier.bias)
    
    def _freeze_model(self, model: nn.Module):
        """Freeze all parameters of a model."""
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    
    def forward(
        self, 
        pixel_values: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        output_attentions: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass through all three models.
        
        Args:
            pixel_values: Input video tensor
            labels: Ground truth labels (optional)
            output_hidden_states: Whether to output hidden states
            output_attentions: Whether to output attention weights
            
        Returns:
            Dictionary containing outputs from all three models
        """
        # Ensure all models are on the same device as inputs
        target_device = pixel_values.device
        # Move teacher and assistant if needed (student is handled by HF Trainer)
        if next(self.teacher_model.parameters()).device != target_device:
            self.teacher_model.to(target_device)
        if hasattr(self, 'assistant_model') and next(self.assistant_model.parameters()).device != target_device:
            self.assistant_model.to(target_device)
        if next(self.student_model.parameters()).device != target_device:
            self.student_model.to(target_device)

        # Student forward pass (with gradients)
        student_outputs = self.student_model(
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                pixel_values=pixel_values,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions
            )
        
        # Assistant forward pass (no gradients)
        with torch.no_grad():
            assistant_outputs = self.assistant_model(
                pixel_values=pixel_values,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions
            )
        
        return {
            'student': student_outputs,
            'teacher': teacher_outputs,
            'assistant': assistant_outputs
        }
    
    def get_student_model(self) -> VideoMAEForVideoClassification:
        """Get the student model for training/inference."""
        return self.student_model
    
    def save_student_model(self, save_path: str):
        """Save the fine-tuned student model."""
        self.student_model.save_pretrained(save_path)
        self.image_processor.save_pretrained(save_path)
        logger.info(f"Student model saved to {save_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all models in the framework."""
        # Determine assistant model name
        if hasattr(self.config, 'assistant_model_name') and self.config.assistant_model_name:
            assistant_name = self.config.assistant_model_name
        elif hasattr(self.config, 'assistant_model_path') and self.config.assistant_model_path:
            assistant_name = 'SSV2_pretrained'
        else:
            assistant_name = 'teacher_fallback'
            
        return {
            'teacher': {
                'name': self.config.teacher_model_name,
                'num_parameters': sum(p.numel() for p in self.teacher_model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.teacher_model.parameters() if p.requires_grad)
            },
            'assistant': {
                'name': assistant_name,
                'num_parameters': sum(p.numel() for p in self.assistant_model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.assistant_model.parameters() if p.requires_grad)
            },
            'student': {
                'name': self.config.student_model_name,
                'num_parameters': sum(p.numel() for p in self.student_model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
            }
        }


# Helper functions for notebook compatibility
def load_teacher_model(config: TriModelConfig) -> VideoMAEForVideoClassification:
    """
    Load and initialize the teacher model.
    
    Args:
        config: Tri-model configuration
        
    Returns:
        Initialized teacher model
    """
    logger.info(f"Loading teacher model: {config.teacher_model_name}")
    
    model = VideoMAEForVideoClassification.from_pretrained(
        config.teacher_model_name,
        num_labels=config.num_labels if hasattr(config, 'num_labels') else 2,
        ignore_mismatched_sizes=True
    )
    
    # Freeze the model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    logger.info("Teacher model loaded and frozen")
    return model


def load_assistant_model(ssv2_config: SSV2ModelConfig, tri_config: TriModelConfig) -> VideoMAEForVideoClassification:
    """
    Load and initialize the assistant model.
    
    Args:
        ssv2_config: SSV2 model configuration
        tri_config: Tri-model configuration
        
    Returns:
        Initialized assistant model
    """
    # First check if we have a HuggingFace assistant model specified
    if hasattr(tri_config, 'assistant_model_name') and tri_config.assistant_model_name:
        logger.info(f"Loading assistant model from HuggingFace: {tri_config.assistant_model_name}")
        try:
            model = VideoMAEForVideoClassification.from_pretrained(
                tri_config.assistant_model_name,
                num_labels=tri_config.num_labels if hasattr(tri_config, 'num_labels') else 2,
                ignore_mismatched_sizes=True
            )
            logger.info(f"Successfully loaded assistant model from HuggingFace")
        except Exception as e:
            logger.warning(f"Could not load HuggingFace assistant model: {e}")
            logger.info("Falling back to teacher model as assistant")
            
            # Fallback to teacher model
            model = VideoMAEForVideoClassification.from_pretrained(
                tri_config.teacher_model_name,
                num_labels=tri_config.num_labels if hasattr(tri_config, 'num_labels') else 2,
                ignore_mismatched_sizes=True
            )
    # Otherwise try to load from SSV2 local path
    elif hasattr(tri_config, 'assistant_model_path') and tri_config.assistant_model_path:
        logger.info("Loading assistant model from SSV2 local path")
        try:
            ssv2_loader = SSV2ModelLoader(ssv2_config)
            model = ssv2_loader.load_model(
                backbone="ViT-B",
                use_finetuned=True,
                num_labels=ssv2_config.num_classes
            )
            
            # Adapt classifier if needed
            if hasattr(tri_config, 'num_labels') and tri_config.num_labels != ssv2_config.num_classes:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, tri_config.num_labels)
                nn.init.normal_(model.classifier.weight, std=0.02)
                nn.init.zeros_(model.classifier.bias)
            
        except Exception as e:
            logger.warning(f"Could not load SSV2 assistant model: {e}")
            logger.info("Falling back to teacher model as assistant")
            
            # Fallback to teacher model
            model = VideoMAEForVideoClassification.from_pretrained(
                tri_config.teacher_model_name,
                num_labels=tri_config.num_labels if hasattr(tri_config, 'num_labels') else 2,
                ignore_mismatched_sizes=True
            )
    else:
        logger.info("No assistant model specified, using teacher as assistant")
        model = VideoMAEForVideoClassification.from_pretrained(
            tri_config.teacher_model_name,
            num_labels=tri_config.num_labels if hasattr(tri_config, 'num_labels') else 2,
            ignore_mismatched_sizes=True
        )
    
    # Freeze the model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    logger.info("Assistant model loaded and frozen")
    return model


def load_student_model(config: TriModelConfig, label2id: dict = None, id2label: dict = None):
    """
    Enhanced student model loading with automatic label alignment.
    """
    if config.use_pretrained_student:
        if not config.pretrained_student_model:
            logger.error("[ERROR] use_pretrained_student is True but pretrained_student_model is not specified")
            raise ValueError("pretrained_student_model must be provided when use_pretrained_student=True")
            
        logger.info(f"[OK] Loading PRE-TRAINED student model with label alignment: {config.pretrained_student_model}")
        try:
            # Use the enhanced function with label alignment
            student_model = load_pretrained_student_model(
                model_name=config.pretrained_student_model,
                num_labels=config.num_labels,
                label2id=label2id,
                id2label=id2label
            )
            logger.info("[OK] Successfully loaded pre-trained student model with label alignment.")
            return student_model
        except Exception as e:
            logger.error(f"[ERROR] Failed to load pre-trained student model: {e}")
            logger.error("Failing over to creating a new tiny student model from scratch.")

    # Rest of the function remains the same...
    logger.info(f"Loading student model: {config.student_model_name}")

    if config.use_tiny_student:
        # Create a tiny student model from the teacher's configuration
        from transformers import VideoMAEConfig
        
        # Load base configuration and modify it for smaller model
        base_config = VideoMAEConfig.from_pretrained(config.student_model_name)
        
        # Get student architecture parameters from config
        student_hidden_size = getattr(config, 'student_hidden_size', 384)
        student_num_attention_heads = getattr(config, 'student_num_attention_heads', 6)
        
        # Create tiny model configuration
        tiny_config = VideoMAEConfig(
            # Core architecture - much smaller
            hidden_size=student_hidden_size,  # Configurable hidden size
            intermediate_size=student_hidden_size * 4,  # 4x hidden size as typical
            num_hidden_layers=4,  # Much fewer layers
            num_attention_heads=student_num_attention_heads,  # Configurable attention heads
            
            # Keep video-specific parameters
            image_size=base_config.image_size,
            patch_size=base_config.patch_size,
            num_channels=base_config.num_channels,
            tubelet_size=base_config.tubelet_size,
            num_frames=base_config.num_frames,
            
            # Classification head
            num_labels=config.num_labels if hasattr(config, 'num_labels') else 2,
            
            # Other parameters
            hidden_act=base_config.hidden_act,
            hidden_dropout_prob=0.1,  # Slightly higher dropout for regularization
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=base_config.layer_norm_eps,
            qkv_bias=base_config.qkv_bias,
        )
        
        # Create model with tiny configuration
        student_model = VideoMAEForVideoClassification(tiny_config)
        
        # Initialize weights properly
        student_model.apply(student_model._init_weights)
        
        # Log model size comparison
        total_params = sum(p.numel() for p in student_model.parameters())
        trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
        
        logger.info(f"Tiny student model created:")
        logger.info(f"  - Layers: {tiny_config.num_hidden_layers} (vs 12 in teacher)")
        logger.info(f"  - Hidden size: {tiny_config.hidden_size} (vs 768 in teacher)")
        logger.info(f"  - Attention heads: {tiny_config.num_attention_heads} (vs 12 in teacher)")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Parameter reduction: ~{((87000000 - total_params) / 87000000 * 100):.1f}% vs full VideoMAE")
        
        return student_model
    else:
        # Load the full student model
        student_model = VideoMAEForVideoClassification.from_pretrained(
            config.student_model_name,
            num_labels=config.num_labels if hasattr(config, 'num_labels') else 2,
            ignore_mismatched_sizes=True
        )
        logger.info("Full-size student model loaded")
    
    # Student model should be trainable
    student_model.train()
    
    logger.info("Student model loaded (trainable)")
    return student_model

def load_pretrained_student_model(
    model_name: str,
    num_labels: int = 2,
    label2id: dict = None,
    id2label: dict = None
) -> VideoMAEForVideoClassification:
    """
    Load pretrained student model with automatic label alignment.
    """
    try:
        logger.info(f"Loading pretrained student model: {model_name}")
        
        # First, get the pretrained model's configuration
        pretrained_config = VideoMAEConfig.from_pretrained(model_name)
        pretrained_num_labels = pretrained_config.num_labels
        pretrained_label2id = getattr(pretrained_config, 'label2id', None)
        pretrained_id2label = getattr(pretrained_config, 'id2label', None)
        
        logger.info(f"Pretrained model has {pretrained_num_labels} labels")
        logger.info(f"Target dataset has {num_labels} labels")
        
        # FIXED: Check for exact label mapping match (not just count)
        if (pretrained_num_labels == num_labels and 
            pretrained_label2id and label2id and
            _labels_actually_match(pretrained_label2id, label2id)):
            logger.info("✅ Perfect label mappings match - loading directly")
            student_model = VideoMAEForVideoClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes=False  # Should work without issues
            )
            
        # Check if we can create a label mapping
        elif (pretrained_label2id and label2id and 
              _can_align_labels(pretrained_label2id, label2id)):
            logger.info("✅ Label alignment possible - creating mapping")
            student_model = _load_with_label_alignment(
                model_name, pretrained_label2id, label2id, id2label
            )
            
        # Handle subset case (target labels are subset of pretrained)
        elif (pretrained_label2id and label2id and 
              _is_label_subset(label2id, pretrained_label2id)):
            logger.info("✅ Target labels are subset of pretrained - extracting relevant classes")
            student_model = _load_with_label_subset(
                model_name, pretrained_label2id, label2id, id2label
            )
            
        else:
            # Fall back to label alignment with warning
            logger.warning(f"❌ Label mismatch detected: {pretrained_num_labels} vs {num_labels}")
            logger.warning("Applying label alignment to preserve as much pretrained knowledge as possible")
            
            if pretrained_label2id and label2id:
                student_model = _load_with_label_alignment(
                    model_name, pretrained_label2id, label2id, id2label
                )
            else:
                logger.warning("No label mappings available - reinitializing classifier head")
                student_model = VideoMAEForVideoClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    label2id=label2id,
                    id2label=id2label,
                    ignore_mismatched_sizes=True
                )
                _reinitialize_classifier_head(student_model, num_labels)
            
        # Ensure all parameters are trainable
        for param in student_model.parameters():
            param.requires_grad = True
            
        logger.info(f"✅ Student model loaded successfully")
        logger.info(f"   Parameters: {sum(p.numel() for p in student_model.parameters()):,}")
        logger.info(f"   Trainable: {sum(p.numel() for p in student_model.parameters() if p.requires_grad):,}")
        
        return student_model
        
    except Exception as e:
        logger.error(f"Failed to load pretrained student model: {e}")
        logger.info("Creating tiny student from scratch as fallback")
        
        # Fallback to creating tiny model
        from transformers import VideoMAEConfig
        config = VideoMAEConfig(
            num_hidden_layers=4,
            hidden_size=384,
            num_attention_heads=6,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label
        )
        return VideoMAEForVideoClassification(config)

def _labels_actually_match(pretrained_label2id: dict, target_label2id: dict) -> bool:
    """Check if label mappings actually match (not just counts)."""
    if not pretrained_label2id or not target_label2id:
        return False
        
    # Must have same number of labels
    if len(pretrained_label2id) != len(target_label2id):
        return False
        
    # Check if all target labels exist in pretrained with same IDs
    for label, target_id in target_label2id.items():
        if label not in pretrained_label2id:
            logger.info(f"Label '{label}' not found in pretrained model")
            return False
        if pretrained_label2id[label] != target_id:
            logger.info(f"Label '{label}': pretrained ID {pretrained_label2id[label]} != target ID {target_id}")
            return False
    
    logger.info("All label mappings match perfectly!")
    return True

def _can_align_labels(pretrained_label2id: dict, target_label2id: dict) -> bool:
    """Check if labels can be aligned between pretrained and target."""
    pretrained_labels = set(pretrained_label2id.keys())
    target_labels = set(target_label2id.keys())
    
    # Check if there's significant overlap (>50%)
    overlap = pretrained_labels.intersection(target_labels)
    overlap_ratio = len(overlap) / len(target_labels)
    
    logger.info(f"Label overlap: {len(overlap)}/{len(target_labels)} ({overlap_ratio:.2%})")
    return overlap_ratio > 0.5

def _is_label_subset(target_label2id: dict, pretrained_label2id: dict) -> bool:
    """Check if target labels are a subset of pretrained labels."""
    target_labels = set(target_label2id.keys())
    pretrained_labels = set(pretrained_label2id.keys())
    
    is_subset = target_labels.issubset(pretrained_labels)
    if is_subset:
        logger.info(f"Target labels are subset of pretrained labels")
    return is_subset

def _load_with_label_alignment(model_name: str, pretrained_label2id: dict, 
                              target_label2id: dict, target_id2label: dict):
    """Load model and align labels by remapping the classifier."""
    # Load with original labels first
    model = VideoMAEForVideoClassification.from_pretrained(model_name)
    
    # Create mapping from pretrained indices to target indices
    index_mapping = {}
    for target_label, target_idx in target_label2id.items():
        if target_label in pretrained_label2id:
            pretrained_idx = pretrained_label2id[target_label]
            index_mapping[pretrained_idx] = target_idx
    
    # Extract and remap classifier weights
    old_classifier = model.classifier
    new_classifier = nn.Linear(old_classifier.in_features, len(target_label2id))
    
    # Initialize new classifier
    nn.init.normal_(new_classifier.weight, std=0.02)
    nn.init.zeros_(new_classifier.bias)
    
    # Copy weights for aligned labels
    with torch.no_grad():
        for old_idx, new_idx in index_mapping.items():
            new_classifier.weight[new_idx] = old_classifier.weight[old_idx]
            new_classifier.bias[new_idx] = old_classifier.bias[old_idx]
    
    model.classifier = new_classifier
    model.config.num_labels = len(target_label2id)
    model.config.label2id = target_label2id
    model.config.id2label = target_id2label
    
    logger.info(f"Aligned {len(index_mapping)} labels from pretrained model")
    return model

def _load_with_label_subset(model_name: str, pretrained_label2id: dict,
                           target_label2id: dict, target_id2label: dict):
    """Load model and extract subset of labels."""
    # Load with original labels
    model = VideoMAEForVideoClassification.from_pretrained(model_name)
    
    # Create new classifier with target size
    old_classifier = model.classifier
    new_classifier = nn.Linear(old_classifier.in_features, len(target_label2id))
    
    # Extract weights for target labels
    with torch.no_grad():
        for target_label, target_idx in target_label2id.items():
            if target_label in pretrained_label2id:
                pretrained_idx = pretrained_label2id[target_label]
                new_classifier.weight[target_idx] = old_classifier.weight[pretrained_idx]
                new_classifier.bias[target_idx] = old_classifier.bias[pretrained_idx]
            else:
                # Initialize randomly if label not found
                nn.init.normal_(new_classifier.weight[target_idx], std=0.02)
                nn.init.zeros_(new_classifier.bias[target_idx])
    
    model.classifier = new_classifier
    model.config.num_labels = len(target_label2id)
    model.config.label2id = target_label2id
    model.config.id2label = target_id2label
    
    logger.info(f"Extracted subset of {len(target_label2id)} labels from pretrained model")
    return model

def _reinitialize_classifier_head(model: VideoMAEForVideoClassification, num_labels: int):
    """Properly reinitialize the classifier head."""
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_labels)
    
    # Use proper initialization
    nn.init.normal_(model.classifier.weight, std=0.02)
    nn.init.zeros_(model.classifier.bias)
    
    logger.info(f"Reinitialized classifier head for {num_labels} classes")

# Alias for backward compatibility
TriModelOrchestrator = TriModelDistillationFramework
