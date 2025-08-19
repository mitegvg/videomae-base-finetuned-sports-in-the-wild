# Tri-Model Asymmetric Distillation Framework

This framework implements the Tri-model Asymmetric Distillation approach for efficient domain-specific video classification, based on the research paper "Tri-Model Asymmetric Distillation Framework for Efficient Domain-specific Video Classification".

## Overview

The framework uses three models to achieve superior performance on domain-specific video classification tasks:

1. **Teacher Model**: VideoMAE base pretrained model (frozen)
2. **Assistant Model**: SSV2 pretrained model from AMD MODEL_ZOO (frozen)
3. **Student Model**: Target model for fine-tuning on the specific domain

### Key Features

- **Asymmetric Knowledge Transfer**: Leverages complementary knowledge from both teacher and assistant models
- **Feature Distillation**: Transfers intermediate representations from teacher and assistant to student
- **Attention Distillation**: Aligns attention patterns between models
- **Flexible Architecture**: Supports different backbone sizes (ViT-S, ViT-B)
- **Domain Adaptation**: Efficiently adapts pretrained models to new domains

## Installation

### 1. Clone and Setup Environment

```bash
cd your-project-directory
pip install -r tri_model_distillation/requirements.txt
```

### 2. Install Additional Dependencies

```bash
# If you encounter issues with pytorchvideo
pip install pytorchvideo --no-deps
pip install av iopath

# For Google Drive downloads
pip install gdown
```

## Usage

### Quick Start with Jupyter Notebook

The easiest way to use the framework is through the provided Jupyter notebook:

```python
# In a Jupyter notebook
from tri_model_distillation import *

# Basic training configuration
config = {
    "dataset_root": "processed_dataset",
    "output_dir": "tri_model_distilled_videomae",
    "learning_rate": 5e-5,
    "batch_size": 4,
    "num_epochs": 10,
    "feature_distillation_weight": 1.0,
    "assistant_feature_weight": 0.8,
}

# Train the model
framework, trainer, results = train_tri_model_distillation(**config)
```

### Command Line Usage

```bash
cd your-project-directory
python -m tri_model_distillation.train
```

### Programmatic Usage

```python
from tri_model_distillation.config import TriModelConfig
from tri_model_distillation.models import TriModelDistillationFramework
from tri_model_distillation.trainer import TriModelDistillationTrainer
from tri_model_distillation.utils import load_label_mappings, create_data_loaders

# Load your dataset
label2id, id2label = load_label_mappings("processed_dataset")

# Configure the framework
config = TriModelConfig(
    teacher_model_name="MCG-NJU/videomae-base",
    student_model_name="MCG-NJU/videomae-base",
    feature_distillation_weight=1.0,
    assistant_feature_weight=0.8,
    temperature=4.0
)

# Initialize framework
framework = TriModelDistillationFramework(
    config=config,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(
    dataset_root="processed_dataset",
    image_processor=framework.image_processor,
    label2id=label2id,
    batch_size=4
)

# Initialize and run trainer
trainer = TriModelDistillationTrainer(
    framework=framework,
    distillation_config=config,
    args=training_args,
    train_dataset=train_loader.dataset,
    eval_dataset=val_loader.dataset
)

trainer.train()
```

## Configuration Options

### TriModelConfig Parameters

- `teacher_model_name`: HuggingFace model name for teacher (default: "MCG-NJU/videomae-base")
- `assistant_model_path`: Path to SSV2 pretrained model (auto-downloaded if None)
- `student_model_name`: HuggingFace model name for student (default: "MCG-NJU/videomae-base")
- `feature_distillation_weight`: Weight for feature distillation loss (default: 1.0)
- `attention_distillation_weight`: Weight for attention distillation loss (default: 0.5)
- `teacher_feature_weight`: Weight for teacher features (default: 1.0)
- `assistant_feature_weight`: Weight for assistant features (default: 0.8)
- `temperature`: Temperature for knowledge distillation (default: 4.0)
- `hidden_layers_to_align`: List of layer indices to align (default: [-1, -2, -3])

### Training Parameters

- `learning_rate`: Learning rate for optimization (default: 5e-5)
- `batch_size`: Batch size for training (default: 4)
- `num_epochs`: Number of training epochs (default: 10)
- `warmup_ratio`: Warmup ratio for learning rate scheduler (default: 0.1)
- `eval_steps`: Steps between evaluations (default: 100)
- `save_steps`: Steps between model saves (default: 200)

## Architecture Details

### Models

1. **Teacher Model**: 
   - VideoMAE base model pretrained on general video data
   - Provides strong general video understanding capabilities
   - Frozen during training

2. **Assistant Model**:
   - SSV2 pretrained model from AMD MODEL_ZOO
   - Provides domain-specific temporal understanding
   - Automatically downloaded from Google Drive
   - Frozen during training

3. **Student Model**:
   - Same architecture as teacher but trainable
   - Learns from both teacher and assistant through distillation
   - Target model for deployment

### Loss Function

The total loss combines multiple components:

```
Total Loss = α * Classification Loss + 
             β * Feature Distillation Loss + 
             γ * Attention Distillation Loss + 
             δ * Asymmetric Knowledge Loss
```

Where:
- **Classification Loss**: Standard cross-entropy loss with ground truth
- **Feature Distillation Loss**: MSE between student and teacher/assistant hidden states
- **Attention Distillation Loss**: MSE between attention patterns
- **Asymmetric Knowledge Loss**: Encourages diversity between teacher and assistant

## Dataset Format

The framework expects your dataset to be preprocessed with the following structure:

```
processed_dataset/
├── train.csv
├── val.csv
├── test.csv
└── videos/
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

CSV files should contain:
```
videos/video1.mp4 A
videos/video2.mp4 B-1
videos/video3.mp4 A
...
```

## Model Downloads

The framework automatically downloads SSV2 pretrained models from the AMD MODEL_ZOO:

- **ViT-S**: Smaller, faster model
- **ViT-B**: Larger, more accurate model

Models are cached locally in the `ssv2_models/` directory.

## Performance Tips

1. **GPU Memory**: Use smaller batch sizes (2-4) for ViT-B models
2. **Training Speed**: Start with ViT-S for faster experimentation
3. **Convergence**: Monitor all loss components during training
4. **Data Loading**: Reduce `num_workers` if encountering issues

## Evaluation Metrics

The framework provides comprehensive evaluation:

- **Accuracy**: Overall classification accuracy
- **Per-class Accuracy**: Accuracy for each class
- **Loss Components**: Individual loss values for debugging
- **Parameter Efficiency**: Comparison of model sizes

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```python
   # Reduce batch size
   config["batch_size"] = 2
   
   # Or use gradient accumulation
   training_args.gradient_accumulation_steps = 2
   ```

2. **Model Download Fails**:
   ```python
   # Manual download from MODEL_ZOO links
   # Place in ssv2_models/ directory
   ```

3. **Dataset Loading Errors**:
   ```python
   # Check CSV format and video paths
   # Ensure videos are readable by OpenCV
   ```

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{tri_model_distillation_2024,
  title={Tri-Model Asymmetric Distillation Framework for Efficient Domain-specific Video Classification},
  author={Your Name},
  journal={Your Conference/Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- AMD framework: [Asymmetric Masked Distillation](https://github.com/MCG-NJU/AMD)
- VideoMAE: [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://github.com/MCG-NJU/VideoMAE)
- HuggingFace Transformers: [Transformers](https://github.com/huggingface/transformers)
