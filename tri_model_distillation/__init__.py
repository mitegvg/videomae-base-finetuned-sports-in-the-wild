"""
Tri-Model Asymmetric Distillation Framework

A framework for efficient domain-specific video classification using knowledge distillation
from multiple teacher models to a smaller student model.

This package provides:
- Feature alignment between models with different architectures
- Asymmetric knowledge transfer
- Memory-efficient training
- Comprehensive evaluation tools
"""

__version__ = "0.1.0"

# Core components
from .config import TriModelConfig, SSV2ModelConfig
from .models import (
    TriModelDistillationFramework,
    load_teacher_model,
    load_assistant_model,
    load_student_model,
    load_pretrained_student_model
)
from .losses import TriModelDistillationLoss
from .trainer import TriModelDistillationTrainer, compute_video_classification_metrics

# Utilities
from .utils import (
    load_dataset_from_csv,
    preprocess_video_data,
    compute_metrics,
    create_binary_label_mapping,
    make_metrics_fn
)

# New modules
from .analysis import (
    analyze_model_architecture,
    calculate_efficiency_metrics,
    print_model_comparison
)
from .visualization import (
    visualize_model_comparison,
    visualize_feature_alignment,
    visualize_alignment_strategy,
    visualize_performance_comparison
)
from .testing import (
    test_attention_alignment,
    test_feature_alignment
)
from .inference import (
    predict_video_class,
    batch_inference,
    create_demo_inference
)
from .evaluation import (
    robust_compute_metrics,
    compare_model_performance,
    calculate_efficiency_performance_tradeoff,
    print_performance_summary,
    save_evaluation_results
)
from .data import (
    process_dataset_with_memory_optimization,
    create_binary_label_mapping,
    prepare_dataloaders
)

# Define what's available for import with "from tri_model_distillation import *"
__all__ = [
    # Core components
    "TriModelConfig", "SSV2ModelConfig",
    "TriModelDistillationFramework",
    "load_teacher_model", "load_assistant_model", "load_student_model", "load_pretrained_student_model",
    "TriModelDistillationLoss",
    "TriModelDistillationTrainer", "compute_video_classification_metrics",
    
    # Utilities
    "load_dataset_from_csv", "preprocess_video_data", "compute_metrics", "create_binary_label_mapping", "make_metrics_fn",
    
    # New modules
    "analyze_model_architecture", "calculate_efficiency_metrics", "print_model_comparison",
    "visualize_model_comparison", "visualize_feature_alignment", "visualize_alignment_strategy", "visualize_performance_comparison",
    "test_attention_alignment", "test_feature_alignment",
    "predict_video_class", "batch_inference", "create_demo_inference",
    "robust_compute_metrics", "compare_model_performance", "calculate_efficiency_performance_tradeoff", 
    "print_performance_summary", "save_evaluation_results",
    "process_dataset_with_memory_optimization", "prepare_dataloaders"
]