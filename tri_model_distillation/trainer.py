"""
Tri-Model Asymmetric Distillation Trainer

This module implements the custom trainer for the tri-model distillation framework,
integrating with HuggingFace Trainer while supporting multiple models and custom loss functions.
"""

import os
import torch
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalLoopOutput, denumpify_detensorize
import numpy as np

# Handle both relative and absolute imports
try:
    from .models import TriModelDistillationFramework
    from .losses import TriModelDistillationLoss
    from .config import TriModelConfig
except ImportError:
    from models import TriModelDistillationFramework
    from losses import TriModelDistillationLoss
    from config import TriModelConfig

logger = logging.getLogger(__name__)


class TriModelDistillationTrainer(Trainer):
    """
    Custom trainer for tri-model asymmetric distillation.
    
    Extends HuggingFace Trainer to support:
    1. Multiple models (teacher, assistant, student)
    2. Custom distillation loss functions
    3. Asymmetric knowledge transfer
    """
    
    def __init__(
        self,
        framework: TriModelDistillationFramework,
        distillation_config: TriModelConfig,
        args: TrainingArguments,
        **kwargs
    ):
        """Initialize trainer and configure distillation components."""
        # Framework & config
        self.framework = framework
        self.distillation_config = distillation_config

        dc = distillation_config  # shorthand
        # Construct loss fn with core weights
        self.distillation_loss_fn = TriModelDistillationLoss(
            feature_distillation_weight=dc.feature_distillation_weight,
            attention_distillation_weight=dc.attention_distillation_weight,
            logits_distillation_weight=dc.logits_distillation_weight,
            classification_loss_weight=dc.classification_loss_weight,
            teacher_feature_weight=dc.teacher_feature_weight,
            assistant_feature_weight=dc.assistant_feature_weight,
            teacher_logits_weight=dc.teacher_logits_weight,
            assistant_logits_weight=dc.assistant_logits_weight,
            teacher_attention_weight=getattr(dc, 'teacher_attention_weight', None),
            assistant_attention_weight=getattr(dc, 'assistant_attention_weight', None),
            temperature=dc.temperature,
            logits_temperature=dc.logits_temperature,
            hidden_layers_to_align=dc.hidden_layers_to_align,
            classification_type=dc.classification_type,
        )
        # Advanced settings
        cfg = dc
        dlf = self.distillation_loss_fn
        dlf.enable_layer_fusion = getattr(cfg, 'enable_layer_fusion', False)
        dlf.layer_fusion_mode = getattr(cfg, 'layer_fusion_mode', 'attention')
        dlf.layer_fusion_teacher_layers = getattr(cfg, 'layer_fusion_teacher_layers', [])
        dlf.layer_fusion_assistant_layers = getattr(cfg, 'layer_fusion_assistant_layers', [])
        dlf.fusion_source = getattr(cfg, 'fusion_source', 'teacher')
        dlf.fusion_source_weighting = getattr(cfg, 'fusion_source_weighting', 'learned')
        dlf.fusion_teacher_weight = getattr(cfg, 'fusion_teacher_weight', 0.5)
        dlf.fusion_assistant_weight = getattr(cfg, 'fusion_assistant_weight', 0.5)
        dlf.fusion_projection_dim = getattr(cfg, 'fusion_projection_dim', None)
        dlf.temporal_delta_distillation_weight = getattr(cfg, 'temporal_delta_distillation_weight', 0.0)
        dlf.temporal_delta_layers = getattr(cfg, 'temporal_delta_layers', [-1])
        dlf.attention_head_squeeze = getattr(cfg, 'attention_head_squeeze', False)
        dlf.attention_head_squeeze_mode = getattr(cfg, 'attention_head_squeeze_mode', 'learned')
        dlf.head_squeeze_ortho_weight = getattr(cfg, 'head_squeeze_ortho_weight', 1e-3)
        # KD gating
        dlf.kd_conf_threshold = getattr(cfg, 'kd_conf_threshold', 0.5)
        dlf.kd_min_keep_ratio = getattr(cfg, 'kd_min_keep_ratio', 0.4)
        dlf.kd_dynamic_lower = getattr(cfg, 'kd_dynamic_lower', True)

        # Alias for backward compatibility
        self.distillation_loss = dlf

        # Init base Trainer on student model
        super().__init__(model=framework.get_student_model(), args=args, **kwargs)
        self._log_active_loss_weights()

    # NEW: concise logging of active loss weights
    def _log_active_loss_weights(self):
        logger.info(
            f"[Distillation Weights] cls={self.distillation_loss_fn.classification_loss_weight} "
            f"feat={self.distillation_loss_fn.feature_distillation_weight} "
            f"attn={self.distillation_loss_fn.attention_distillation_weight} "
            f"logits={self.distillation_loss_fn.logits_distillation_weight} "
            f"T={self.distillation_loss_fn.temperature} "
            f"logits_T={getattr(self.distillation_loss_fn,'logits_temperature', None)}"
        )

    def create_optimizer(self):
        """Create optimizer including projection parameters from loss function."""
        # First, run a dummy forward pass to ensure projections are created
        self._ensure_projections_created()
        
        # Create base optimizer for student model
        super().create_optimizer()
        
        # Add projection parameters to optimizer
        extra_params = [p for p in self.distillation_loss_fn.parameters() if p.requires_grad]
        if extra_params:
            logger.info(f"Adding {len(extra_params)} projection parameters to optimizer")
            self.optimizer.add_param_group({
                "params": extra_params,
                "lr": self.args.learning_rate * 0.1  # Lower LR for projections
            })
    
    def _ensure_projections_created(self):
        """Run a dummy forward pass to create projection layers."""
        if hasattr(self, '_projections_created'):
            return
            
        try:
            # Get a sample from train dataloader
            sample_batch = next(iter(self.get_train_dataloader()))
            sample_batch = self._prepare_inputs(sample_batch)
            
            with torch.no_grad():
                # Run forward pass to trigger projection creation
                outputs = self.framework(
                    pixel_values=sample_batch["pixel_values"][:1],  # Just one sample
                    labels=sample_batch["labels"][:1] if "labels" in sample_batch else None,
                    output_hidden_states=True,
                    output_attentions=True
                )
                
                # Trigger loss computation to create projections
                _ = self.distillation_loss_fn(
                    student_outputs=outputs['student'],
                    teacher_outputs=outputs['teacher'],
                    assistant_outputs=outputs['assistant'],
                    labels=sample_batch["labels"][:1] if "labels" in sample_batch else torch.zeros(1, dtype=torch.long, device=outputs['student'].logits.device)
                )
                
            self._projections_created = True
            logger.info("Projection layers created and ready for training")
            
        except Exception as e:
            logger.warning(f"Could not create projections in advance: {e}")
            self._projections_created = True  # Set anyway to avoid infinite loops
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute tri-model distillation loss.
        Must return a scalar tensor. Only return (loss, outputs) if return_outputs=True.
        """
        # Extract inputs
        pixel_values = inputs["pixel_values"]
        labels = inputs["labels"]

        need_hidden = (
            self.distillation_loss_fn.feature_distillation_weight > 0
            or (self.distillation_config.align_hidden_states and len(self.distillation_config.hidden_layers_to_align) > 0)
        )
        need_attn = (
            self.distillation_loss_fn.attention_distillation_weight > 0
            or self.distillation_config.align_attention_maps
        )

        framework_outputs = self.framework(
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=need_hidden,
            output_attentions=need_attn
        )

        # Expect framework_outputs to contain student/teacher/assistant model outputs
        loss_dict = self.distillation_loss_fn(
            student_outputs=framework_outputs["student"],
            teacher_outputs=framework_outputs.get("teacher"),
            assistant_outputs=framework_outputs.get("assistant"),
            labels=labels
        )

        total_loss = loss_dict["total_loss"]

        # Optional logging of components
        if self.state.global_step % self.args.logging_steps == 0:
            try:
                self.log({
                    "loss_total": float(total_loss.detach()),
                    "loss_cls": float(loss_dict.get("classification_loss", 0.0)),
                    "loss_logits": float(loss_dict.get("logits_distillation_loss", 0.0)),
                    "loss_feat": float(loss_dict.get("feature_distillation_loss", 0.0)),
                    "loss_delta": float(loss_dict.get("temporal_delta_distillation_loss", 0.0)),
                    "loss_attn": float(loss_dict.get("attention_distillation_loss", 0.0)),
                    **({} if not (self.distillation_loss_fn.enable_layer_fusion and getattr(self.distillation_loss_fn, '_fusion_source_logits', None) is not None) else {
                        "fusion_src_w_teacher": float(torch.softmax(self.distillation_loss_fn._fusion_source_logits, dim=0)[0].detach()),
                        "fusion_src_w_assistant": float(torch.softmax(self.distillation_loss_fn._fusion_source_logits, dim=0)[1].detach()),
                    })
                })
            except Exception:
                pass

        if return_outputs:
            return total_loss, framework_outputs["student"]
        return total_loss
    
    def evaluate(
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """Robust evaluation supporting dynamic compute_metrics signature."""
        import inspect
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        student = self.framework.get_student_model(); student.eval()
        need_hidden = self.distillation_loss_fn.feature_distillation_weight > 0
        need_attn = self.distillation_loss_fn.attention_distillation_weight > 0

        total_loss = total_cls = total_feat = total_attn = total_logits = 0.0
        num_samples = 0
        all_logits, all_labels, all_pred = [], [], []

        for inputs in eval_dataloader:
            with torch.no_grad():
                inputs = self._prepare_inputs(inputs)
                pixel_values, labels = inputs["pixel_values"], inputs["labels"]
                outputs = self.framework(
                    pixel_values=pixel_values,
                    labels=labels,
                    output_hidden_states=need_hidden,
                    output_attentions=need_attn
                )
                loss_dict = self.distillation_loss_fn(
                    student_outputs=outputs['student'],
                    teacher_outputs=outputs['teacher'],
                    assistant_outputs=outputs['assistant'],
                    labels=labels
                )
                bsz = labels.size(0)
                num_samples += bsz
                # Accumulate weighted by batch size
                total_loss += float(loss_dict['total_loss']) * bsz
                total_cls += float(loss_dict['classification_loss']) * bsz
                total_feat += float(loss_dict.get('feature_distillation_loss', 0.0)) * bsz
                total_attn += float(loss_dict.get('attention_distillation_loss', 0.0)) * bsz
                total_logits += float(loss_dict.get('logits_distillation_loss', 0.0)) * bsz

                logits_np = outputs['student'].logits.detach().cpu().numpy()
                all_logits.append(logits_np)
                all_labels.append(labels.cpu().numpy())
                all_pred.append(logits_np.argmax(axis=-1))

        import numpy as np
        all_logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0,))
        all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,))
        all_pred = np.concatenate(all_pred, axis=0) if all_pred else np.zeros((0,))

        denom = max(num_samples, 1)
        eval_metrics = {
            f"{metric_key_prefix}_loss": total_loss / denom,
            f"{metric_key_prefix}_classification_loss": total_cls / denom,
            f"{metric_key_prefix}_feature_distillation_loss": total_feat / denom,
            f"{metric_key_prefix}_attention_distillation_loss": total_attn / denom,
            f"{metric_key_prefix}_logits_distillation_loss": total_logits / denom,
            f"{metric_key_prefix}_accuracy": float(np.mean(all_pred == all_labels)) if num_samples else 0.0,
        }

        # Optional external metrics
        if self.compute_metrics and num_samples:
            try:
                sig = inspect.signature(self.compute_metrics)
                if 'classification_type' in sig.parameters:
                    extra = self.compute_metrics((all_logits, all_labels), classification_type=self.distillation_config.classification_type)
                else:
                    extra = self.compute_metrics((all_logits, all_labels))
                for k, v in extra.items():
                    eval_metrics[f"{metric_key_prefix}_{k}"] = v
            except Exception as e:
                logger.warning(f"compute_metrics failed: {e}")

        try:
            if metric_key_prefix == 'eval':
                self._memory_tracker.stop_and_update_metrics(eval_metrics)
        except Exception:
            logger.warning("Memory tracking failed")
            pass

        self.log(eval_metrics)

        # Fire callback so NotebookProgressCallback / custom callbacks can record a row
        # Ensure an 'epoch' field is present (NotebookProgressCallback uses it when eval_strategy='epoch')
        if hasattr(self.state, "epoch") and self.state.epoch is not None:
            eval_metrics["epoch"] = self.state.epoch

        try:
            # Callback handler expects (args, state, control, metrics)
            # Ensure self.control exists (created by Trainer during train(); fall back if absent)
            from transformers.trainer_callback import TrainerControl
            if not hasattr(self, "control") or self.control is None:
                self.control = TrainerControl()
            self.control = self.callback_handler.on_evaluate(
                self.args, self.state, self.control, eval_metrics
            )
        except Exception as e:
            logger.warning(f"on_evaluate callback dispatch failed: {e}")

        return eval_metrics
    
    def prediction_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None):
        """
        Prediction step for evaluation.
        """
        inputs = self._prepare_inputs(inputs)
        need_hidden = self.distillation_loss_fn.feature_distillation_weight > 0
        # keep attentions off for speed unless needed
        need_attn = self.distillation_loss_fn.attention_distillation_weight > 0
        with torch.no_grad():
            pixel_values = inputs["pixel_values"]; labels = inputs.get("labels")
            
            # Forward pass through framework
            outputs = self.framework(
                pixel_values=pixel_values,
                labels=labels,
                output_hidden_states=need_hidden,
                output_attentions=need_attn
            )
            
            student_outputs = outputs['student']
            
            # Compute loss if labels are available
            loss = None
            if labels is not None:
                loss_dict = self.distillation_loss_fn(
                    student_outputs=outputs['student'],
                    teacher_outputs=outputs['teacher'],
                    assistant_outputs=outputs['assistant'],
                    labels=labels
                )
                loss = loss_dict['total_loss']
            
            # Get predictions
            logits = student_outputs.logits
            
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, logits, labels)
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the student model and framework configuration.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # Save student model
        self.framework.save_student_model(output_dir)
        
        # Save framework configuration
        config_path = os.path.join(output_dir, "tri_model_config.json")
        import json
        c = self.distillation_config
        config_dict = {
            "teacher_model_name": c.teacher_model_name,
            "assistant_model_name": getattr(c, 'assistant_model_name', None),
            "student_model_name": c.student_model_name,
            "feature_distillation_weight": c.feature_distillation_weight,
            "attention_distillation_weight": c.attention_distillation_weight,
            "logits_distillation_weight": getattr(c, 'logits_distillation_weight', None),
            "classification_loss_weight": c.classification_loss_weight,
            "teacher_feature_weight": c.teacher_feature_weight,
            "assistant_feature_weight": c.assistant_feature_weight,
            "teacher_logits_weight": getattr(c, 'teacher_logits_weight', None),
            "assistant_logits_weight": getattr(c, 'assistant_logits_weight', None),
            "temperature": c.temperature,
            "logits_temperature": getattr(c, 'logits_temperature', None),
            "hidden_layers_to_align": c.hidden_layers_to_align,
            "classification_type": getattr(c, 'classification_type', None),
            "num_labels": getattr(c, 'num_labels', None),
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Tri-model distillation framework saved to {output_dir}")
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Enhanced logging with distillation-specific metrics.
        Compatible with both old and new transformers versions.
        """
        # Add model information to logs periodically
        if self.state.global_step % (self.args.logging_steps * 10) == 0:
            model_info = self.framework.get_model_info()
            logs.update({
                "model/student_params": model_info['student']['trainable_parameters'],
                "model/teacher_params": model_info['teacher']['num_parameters'],
                "model/assistant_params": model_info['assistant']['num_parameters'],
            })
        
        # Call parent log method with compatible signature
        if start_time is not None:
            try:
                # Try new signature first (transformers >= 4.20)
                super().log(logs, start_time)
            except TypeError:
                # Fall back to old signature
                super().log(logs)
        else:
            super().log(logs)

    # (Removed duplicate evaluate method below to avoid shadowing.)
    
    def prediction_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None):
        """
        Prediction step for evaluation.
        """
        inputs = self._prepare_inputs(inputs)
        need_hidden = self.distillation_loss_fn.feature_distillation_weight > 0
        # keep attentions off for speed unless needed
        need_attn = self.distillation_loss_fn.attention_distillation_weight > 0
        with torch.no_grad():
            pixel_values = inputs["pixel_values"]; labels = inputs.get("labels")
            
            # Forward pass through framework
            outputs = self.framework(
                pixel_values=pixel_values,
                labels=labels,
                output_hidden_states=need_hidden,
                output_attentions=need_attn
            )
            
            student_outputs = outputs['student']
            
            # Compute loss if labels are available
            loss = None
            if labels is not None:
                loss_dict = self.distillation_loss_fn(
                    student_outputs=outputs['student'],
                    teacher_outputs=outputs['teacher'],
                    assistant_outputs=outputs['assistant'],
                    labels=labels
                )
                loss = loss_dict['total_loss']
            
            # Get predictions
            logits = student_outputs.logits
            
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, logits, labels)
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the student model and framework configuration.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # Save student model
        self.framework.save_student_model(output_dir)
        
        # Save framework configuration
        config_path = os.path.join(output_dir, "tri_model_config.json")
        import json
        c = self.distillation_config
        config_dict = {
            "teacher_model_name": c.teacher_model_name,
            "assistant_model_name": getattr(c, 'assistant_model_name', None),
            "student_model_name": c.student_model_name,
            "feature_distillation_weight": c.feature_distillation_weight,
            "attention_distillation_weight": c.attention_distillation_weight,
            "logits_distillation_weight": getattr(c, 'logits_distillation_weight', None),
            "classification_loss_weight": c.classification_loss_weight,
            "teacher_feature_weight": c.teacher_feature_weight,
            "assistant_feature_weight": c.assistant_feature_weight,
            "teacher_logits_weight": getattr(c, 'teacher_logits_weight', None),
            "assistant_logits_weight": getattr(c, 'assistant_logits_weight', None),
            "temperature": c.temperature,
            "logits_temperature": getattr(c, 'logits_temperature', None),
            "hidden_layers_to_align": c.hidden_layers_to_align,
            "classification_type": getattr(c, 'classification_type', None),
            "num_labels": getattr(c, 'num_labels', None),
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Tri-model distillation framework saved to {output_dir}")
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Enhanced logging with distillation-specific metrics.
        Compatible with both old and new transformers versions.
        """
        # Add model information to logs periodically
        if self.state.global_step % (self.args.logging_steps * 10) == 0:
            model_info = self.framework.get_model_info()
            logs.update({
                "model/student_params": model_info['student']['trainable_parameters'],
                "model/teacher_params": model_info['teacher']['num_parameters'],
                "model/assistant_params": model_info['assistant']['num_parameters'],
            })
        
        # Call parent log method with compatible signature
        if start_time is not None:
            try:
                # Try new signature first (transformers >= 4.20)
                super().log(logs, start_time)
            except TypeError:
                # Fall back to old signature
                super().log(logs)
        else:
            super().log(logs)


def compute_video_classification_metrics(eval_pred, classification_type="multiclass") -> Dict[str, float]:
    """
    Comprehensive evaluation metrics for video classification.
    
    Supports binary, multiclass, and multilabel classification with detailed metrics
    including top-k accuracy, per-class metrics, and confusion matrix analysis.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        classification_type: Type of classification ("binary", "multiclass", "multilabel")
        
    Returns:
        Dictionary of computed metrics
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, confusion_matrix,
        hamming_loss, jaccard_score, roc_auc_score, average_precision_score,
        balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef
    )
    
    predictions, labels = eval_pred
    
    # Unwrap tuple if predictions come from model outputs
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Convert tensors to numpy
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    
    # Initialize metrics dictionary
    metrics = {}
    
    try:
        if classification_type == "multilabel":
            # =================================
            # MULTILABEL CLASSIFICATION METRICS
            # =================================
            
            # Apply sigmoid and threshold for multilabel
            if predictions.ndim >= 2:
                # Assume predictions are logits, apply sigmoid
                predictions_prob = 1 / (1 + np.exp(-predictions))  # sigmoid
                predictions_binary = (predictions_prob > 0.5).astype(int)
            else:
                predictions_binary = predictions.astype(int)
                predictions_prob = predictions_binary  # Fallback
            
            # Core multilabel metrics
            exact_match = accuracy_score(labels, predictions_binary)
            hamming = hamming_loss(labels, predictions_binary)
            jaccard = jaccard_score(labels, predictions_binary, average='macro', zero_division=0)
            
            # F1 scores
            f1_macro = precision_recall_fscore_support(labels, predictions_binary, average='macro', zero_division=0)[2]
            f1_micro = precision_recall_fscore_support(labels, predictions_binary, average='micro', zero_division=0)[2]
            f1_weighted = precision_recall_fscore_support(labels, predictions_binary, average='weighted', zero_division=0)[2]
            
            # Precision and Recall
            precision_macro = precision_recall_fscore_support(labels, predictions_binary, average='macro', zero_division=0)[0]
            recall_macro = precision_recall_fscore_support(labels, predictions_binary, average='macro', zero_division=0)[1]
            precision_micro = precision_recall_fscore_support(labels, predictions_binary, average='micro', zero_division=0)[0]
            recall_micro = precision_recall_fscore_support(labels, predictions_binary, average='micro', zero_division=0)[1]
            
            # Add to metrics
            metrics.update({
                "accuracy": exact_match,
                "exact_match_accuracy": exact_match,
                "hamming_loss": hamming,
                "jaccard_score": jaccard,
                "f1_macro": f1_macro,
                "f1_micro": f1_micro,
                "f1_weighted": f1_weighted,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "precision_micro": precision_micro,
                "recall_micro": recall_micro,
            })
            
            # Per-class metrics for multilabel
            num_classes = labels.shape[1] if labels.ndim > 1 else predictions_binary.shape[1]
            class_accuracies = []
            class_f1s = []
            
            for i in range(min(num_classes, 20)):  # Limit to first 20 classes to avoid clutter
                if i < labels.shape[1] and i < predictions_binary.shape[1]:
                    # Per-class accuracy
                    class_acc = accuracy_score(labels[:, i], predictions_binary[:, i])
                    class_accuracies.append(class_acc)
                    
                    # Per-class F1
                    class_f1 = precision_recall_fscore_support(
                        labels[:, i], predictions_binary[:, i], average='binary', zero_division=0
                    )[2]
                    class_f1s.append(class_f1)
                    
                    metrics[f"class_{i}_accuracy"] = class_acc
                    metrics[f"class_{i}_f1"] = class_f1
            
            # Average per-class metrics
            if class_accuracies:
                metrics["avg_class_accuracy"] = np.mean(class_accuracies)
            if class_f1s:
                metrics["avg_class_f1"] = np.mean(class_f1s)
            
            # ROC-AUC for multilabel (if probabilities available)
            try:
                if predictions_prob is not None and predictions_prob.shape == labels.shape:
                    auc_macro = roc_auc_score(labels, predictions_prob, average='macro', multi_class='ovr')
                    auc_micro = roc_auc_score(labels, predictions_prob, average='micro', multi_class='ovr')
                    metrics.update({
                        "roc_auc_macro": auc_macro,
                        "roc_auc_micro": auc_micro,
                    })
            except (ValueError, TypeError):
                pass  # Skip AUC if not applicable
                
        else:
            # =======================================
            # BINARY & MULTICLASS CLASSIFICATION METRICS
            # =======================================
            
            # Convert predictions to class indices
            if predictions.ndim >= 2:
                predictions_class = np.argmax(predictions, axis=-1)
                # Store raw predictions for probability-based metrics
                raw_predictions = predictions
                # Get prediction probabilities
                if predictions.shape[-1] > 1:
                    # Apply softmax for multiclass probabilities
                    exp_preds = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
                    predictions_prob = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
                else:
                    # Binary case - apply sigmoid
                    predictions_prob = 1 / (1 + np.exp(-predictions))
            else:
                predictions_class = predictions.astype(np.int64)
                raw_predictions = None
                predictions_prob = None
            
            # Core accuracy metrics
            accuracy = accuracy_score(labels, predictions_class)
            balanced_acc = balanced_accuracy_score(labels, predictions_class)
            
            # Top-k accuracy calculation
            top1_accuracy = accuracy
            top5_accuracy = 0.0
            
            if raw_predictions is not None and raw_predictions.shape[-1] >= 5:
                # Calculate top-5 accuracy
                top5_preds = np.argsort(raw_predictions, axis=-1)[:, -5:]
                top5_correct = 0
                for i in range(len(labels)):
                    if labels[i] in top5_preds[i]:
                        top5_correct += 1
                top5_accuracy = top5_correct / len(labels)
            
            # Precision, Recall, F1
            if classification_type == "binary" or len(np.unique(labels)) == 2:
                # Binary classification metrics
                precision, recall, f1, support = precision_recall_fscore_support(
                    labels, predictions_class, average='binary', zero_division=0
                )
                
                # Additional binary metrics
                tn, fp, fn, tp = confusion_matrix(labels, predictions_class).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value (same as precision)
                
                # Matthews Correlation Coefficient
                mcc = matthews_corrcoef(labels, predictions_class)
                
                metrics.update({
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "npv": npv,
                    "ppv": ppv,
                    "mcc": mcc,
                })
                
                # ROC-AUC and PR-AUC for binary
                try:
                    if predictions_prob is not None:
                        if predictions_prob.ndim > 1:
                            prob_positive = predictions_prob[:, 1] if predictions_prob.shape[1] > 1 else predictions_prob[:, 0]
                        else:
                            prob_positive = predictions_prob
                        
                        roc_auc = roc_auc_score(labels, prob_positive)
                        pr_auc = average_precision_score(labels, prob_positive)
                        
                        metrics.update({
                            "roc_auc": roc_auc,
                            "pr_auc": pr_auc,
                        })
                except (ValueError, TypeError):
                    pass  # Skip if not applicable
                    
            else:
                # Multiclass classification metrics
                precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                    labels, predictions_class, average='macro', zero_division=0
                )
                precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                    labels, predictions_class, average='micro', zero_division=0
                )
                precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                    labels, predictions_class, average='weighted', zero_division=0
                )
                
                # Set metrics for compatibility
                precision, recall, f1 = precision_macro, recall_macro, f1_macro
                
                metrics.update({
                    "precision_macro": precision_macro,
                    "precision_micro": precision_micro,
                    "precision_weighted": precision_weighted,
                    "recall_macro": recall_macro,
                    "recall_micro": recall_micro,
                    "recall_weighted": recall_weighted,
                    "f1_macro": f1_macro,
                    "f1_micro": f1_micro,
                    "f1_weighted": f1_weighted,
                })
                
                # Multiclass ROC-AUC
                try:
                    if predictions_prob is not None and predictions_prob.shape[1] > 2:
                        roc_auc_ovr = roc_auc_score(labels, predictions_prob, multi_class='ovr', average='macro')
                        roc_auc_ovo = roc_auc_score(labels, predictions_prob, multi_class='ovo', average='macro')
                        
                        metrics.update({
                            "roc_auc_ovr": roc_auc_ovr,
                            "roc_auc_ovo": roc_auc_ovo,
                        })
                except (ValueError, TypeError):
                    pass  # Skip if not applicable
            
            # Cohen's Kappa
            try:
                kappa = cohen_kappa_score(labels, predictions_class)
                metrics["cohen_kappa"] = kappa
            except (ValueError, TypeError):
                pass
            
            # Per-class accuracy and F1
            unique_labels = np.unique(labels)
            per_class_metrics = {}
            class_accuracies = []
            class_f1s = []
            
            for label in unique_labels[:20]:  # Limit to first 20 classes
                mask = labels == label
                if np.sum(mask) > 0:
                    # Per-class accuracy
                    class_accuracy = np.mean(predictions_class[mask] == labels[mask])
                    class_accuracies.append(class_accuracy)
                    per_class_metrics[f"class_{label}_accuracy"] = class_accuracy
                    
                    # Per-class F1
                    try:
                        class_f1 = precision_recall_fscore_support(
                            labels == label, predictions_class == label, 
                            average='binary', zero_division=0
                        )[2]
                        class_f1s.append(class_f1)
                        per_class_metrics[f"class_{label}_f1"] = class_f1
                    except (ValueError, IndexError):
                        per_class_metrics[f"class_{label}_f1"] = 0.0
            
            # Average per-class metrics
            if class_accuracies:
                per_class_metrics["avg_class_accuracy"] = np.mean(class_accuracies)
            if class_f1s:
                per_class_metrics["avg_class_f1"] = np.mean(class_f1s)
            
            # Add all metrics
            metrics.update({
                "accuracy": accuracy,
                "balanced_accuracy": balanced_acc,
                "top1_accuracy": top1_accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                **per_class_metrics
            })
            
            # Add top-5 accuracy if available
            if top5_accuracy > 0:
                metrics["top5_accuracy"] = top5_accuracy
            
            # Confusion matrix analysis
            try:
                cm = confusion_matrix(labels, predictions_class)
                
                # Overall confusion matrix stats
                total_samples = np.sum(cm)
                correct_predictions = np.trace(cm)
                
                metrics.update({
                    "confusion_matrix_trace": correct_predictions,
                    "confusion_matrix_total": total_samples,
                })
                
                # Per-class precision and recall from confusion matrix
                if cm.shape[0] <= 10:  # Only for small number of classes
                    for i in range(cm.shape[0]):
                        if i < len(unique_labels):
                            # Class precision: tp / (tp + fp)
                            class_precision = cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0.0
                            # Class recall: tp / (tp + fn)
                            class_recall = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0.0
                            
                            metrics[f"class_{unique_labels[i]}_precision"] = class_precision
                            metrics[f"class_{unique_labels[i]}_recall"] = class_recall
                            
            except (ValueError, IndexError):
                pass  # Skip confusion matrix analysis if error
    
    except Exception as e:
        # Fallback to basic metrics if advanced metrics fail
        logger.warning(f"Error computing advanced metrics: {e}")
        
        if predictions.ndim >= 2:
            predictions_class = np.argmax(predictions, axis=-1)
        else:
            predictions_class = predictions.astype(np.int64)
        
        # Basic accuracy as fallback
        accuracy = accuracy_score(labels, predictions_class) if len(labels) > 0 else 0.0
        metrics = {"accuracy": accuracy}
    
    # Ensure all metrics are float (not numpy types)
    metrics = {k: float(v) if not isinstance(v, str) else v for k, v in metrics.items()}
    
    # Add metadata
    metrics.update({
        "num_samples": len(labels),
        "num_classes": len(np.unique(labels)) if len(labels) > 0 else 0,
        "classification_type": classification_type,
    })
    
    return metrics


class DistillationMetricsCallback:
    """Callback for logging additional distillation metrics during training."""
    
    def __init__(self, framework: TriModelDistillationFramework):
        self.framework = framework
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Add distillation-specific metrics to logs."""
        if logs is not None and state.global_step % 100 == 0:
            # Add model-specific information
            model_info = self.framework.get_model_info()
            logs.update({
                "distillation/student_trainable_params": model_info['student']['trainable_parameters'],
                "distillation/total_teacher_params": model_info['teacher']['num_parameters'],
                "distillation/total_assistant_params": model_info['assistant']['num_parameters'],
            })
