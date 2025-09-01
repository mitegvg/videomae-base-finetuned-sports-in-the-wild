"""
Main training script for Tri-Model Asymmetric Distillation Framework

This script orchestrates the complete training process using the tri-model
distillation framework for domain-specific video classification.
"""

import os
import torch
import logging
from datetime import datetime
from transformers import TrainingArguments, TrainerCallback, TrainerControl, TrainerState

# Import framework components
from .config import TriModelConfig, SSV2ModelConfig
from .models import TriModelDistillationFramework
from .trainer import TriModelDistillationTrainer, compute_video_classification_metrics
from .utils import (
    setup_logging, load_label_mappings, create_data_loaders,
    print_model_info, save_training_config, evaluate_model_on_dataset
)

logger = logging.getLogger(__name__)


def train_tri_model_distillation(
    dataset_root: str = "processed_dataset",
    output_dir: str = "tri_model_distilled_videomae",
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    num_epochs: int = 10,
    warmup_ratio: float = 0.1,
    eval_steps: int = 100,
    save_steps: int = 200,
    logging_steps: int = 10,
    feature_distillation_weight: float = 1.0,
    attention_distillation_weight: float = 0.5,
    teacher_feature_weight: float = 1.0,
    assistant_feature_weight: float = 0.8,
    temperature: float = 4.0,
    **kwargs
):
    """
    Main training function for tri-model asymmetric distillation.
    
    Args:
        dataset_root: Root directory containing processed dataset
        output_dir: Directory to save trained model and logs
        learning_rate: Learning rate for optimization
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        warmup_ratio: Warmup ratio for learning rate scheduler
        eval_steps: Steps between evaluations
        save_steps: Steps between model saves
        logging_steps: Steps between logging
        feature_distillation_weight: Weight for feature distillation loss
        attention_distillation_weight: Weight for attention distillation loss
        teacher_feature_weight: Weight for teacher features in distillation
        assistant_feature_weight: Weight for assistant features in distillation
        temperature: Temperature for knowledge distillation
        **kwargs: Additional arguments
    """
    
    # Setup
    setup_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"tri_model_distillation_{timestamp}"
    
    logger.info("=" * 60)
    logger.info("Tri-Model Asymmetric Distillation Framework")
    logger.info("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    
    # Load label mappings
    logger.info("Loading label mappings...")
    label2id, id2label = load_label_mappings(dataset_root)
    num_labels = len(label2id)
    
    logger.info(f"Found {num_labels} classes: {list(label2id.keys())}")
    
    # Create configuration
    tri_config = TriModelConfig(
        dataset_root=dataset_root,
        output_dir=output_dir,
        feature_distillation_weight=feature_distillation_weight,
        attention_distillation_weight=attention_distillation_weight,
        teacher_feature_weight=teacher_feature_weight,
        assistant_feature_weight=assistant_feature_weight,
        temperature=temperature,
        **kwargs
    )
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_total_limit=40,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        run_name=run_name,
        report_to=["tensorboard", "wandb"], 
        logging_dir=os.path.join(output_dir, "logs"),
        disable_tqdm=False, 
        seed=42,
    )
    
    # Initialize framework
    logger.info("Initializing tri-model distillation framework...")
    framework = TriModelDistillationFramework(
        config=tri_config,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
    )
    
    # Print model information
    model_info = framework.get_model_info()
    print("\n" + "=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    
    for model_type, info in model_info.items():
        print(f"\n{model_type.upper()} MODEL:")
        print(f"  Name: {info['name']}")
        print(f"  Total parameters: {info['num_parameters']:,}")
        print(f"  Trainable parameters: {info['trainable_parameters']:,}")
        if info['num_parameters'] > 0:
            trainable_ratio = info['trainable_parameters'] / info['num_parameters'] * 100
            print(f"  Trainable ratio: {trainable_ratio:.2f}%")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_root=dataset_root,
        image_processor=framework.image_processor,
        label2id=label2id,
        batch_size=batch_size,
        num_frames=tri_config.num_frames,
        num_workers=2  # Reduce for stability
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = TriModelDistillationTrainer(
        framework=framework,
        distillation_config=tri_config,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        compute_metrics=compute_video_classification_metrics,
    )
    
    class EvalPrintCallback(TrainerCallback):
        def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
            acc = metrics.get("eval_accuracy")
            if acc is not None:
                print(f"[EVAL] step {state.global_step} | eval_accuracy={acc:.4f}")

    trainer.add_callback(EvalPrintCallback())
    
    # Save configuration
    config_save_path = os.path.join(output_dir, "training_config.json")
    os.makedirs(output_dir, exist_ok=True)
    
    config_dict = {
        "tri_model_config": tri_config.__dict__,
        "training_args": training_args.to_dict(),
        "dataset_info": {
            "num_labels": num_labels,
            "label2id": label2id,
            "id2label": id2label,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
        },
        "model_info": model_info,
        "timestamp": timestamp,
    }
    
    save_training_config(config_dict, config_save_path)
    
    # Start training
    logger.info("Starting training...")
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    try:
        train_results = trainer.train()
        
        logger.info("Training completed successfully!")
        print(f"\nTraining results: {train_results}")
        
        # Final evaluation
        logger.info("Performing final evaluation...")
        
        # Evaluate on validation set
        val_metrics = trainer.evaluate(eval_dataset=val_loader.dataset)
        logger.info(f"Final validation metrics: {val_metrics}")
        
        # Evaluate on test set
        test_metrics = trainer.evaluate(
            eval_dataset=test_loader.dataset,
            metric_key_prefix="test"
        )
        logger.info(f"Final test metrics: {test_metrics}")
        
        # Save final model
        trainer.save_model()
        
        # Save final results
        final_results = {
            "train_results": getattr(train_results, 'training_metrics', {}),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
        }
        
        results_path = os.path.join(output_dir, "final_results.json")
        save_training_config(final_results, results_path)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Model saved to: {output_dir}")
        print(f"Final validation accuracy: {val_metrics.get('eval_accuracy', 'N/A'):.4f}")
        print(f"Final test accuracy: {test_metrics.get('test_accuracy', 'N/A'):.4f}")
        print("=" * 60)
        
        print("Last 5 log history entries:")
        for row in trainer.state.log_history[-5:]:
            print(row)
        
        return framework, trainer, final_results
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


def main():
    """Main entry point for training script."""
    
    # Default configuration - can be modified or made configurable
    config = {
        "dataset_root": "processed_dataset",
        "output_dir": "tri_model_distilled_videomae",
        "learning_rate": 5e-5,
        "batch_size": 4,  # Small batch size for GPU memory
        "num_epochs": 10,
        "warmup_ratio": 0.1,
        "eval_steps": 100,
        "save_steps": 200,
        "logging_steps": 10,
        "feature_distillation_weight": 1.0,
        "attention_distillation_weight": 0.5,
        "teacher_feature_weight": 1.0,
        "assistant_feature_weight": 0.8,
        "temperature": 4.0,
    }
    
    # Train the model
    framework, trainer, results = train_tri_model_distillation(**config)
    
    return framework, trainer, results


if __name__ == "__main__":
    main()
