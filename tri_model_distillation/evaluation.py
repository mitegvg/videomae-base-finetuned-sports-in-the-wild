"""
Evaluation Utilities for Tri-Model Distillation Framework

This module provides functions for evaluating and comparing model performance.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Optional, Any, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def robust_compute_metrics(eval_tuple):
    """
    Robust compute metrics function that handles various input formats.
    
    Args:
        eval_tuple: Tuple of (predictions, labels)
        
    Returns:
        Dictionary with computed metrics
    """
    predictions, labels = eval_tuple
    
    # Convert to numpy arrays
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    if isinstance(labels, list):
        labels = np.array(labels)
    
    # Handle predictions
    if len(predictions.shape) == 1:
        # Already class indices
        pred_classes = predictions
    elif predictions.shape[1] == 1:
        # Single column of predictions
        pred_classes = predictions.flatten()
    else:
        # Logits - take argmax
        pred_classes = np.argmax(predictions, axis=1)
    
    # Handle labels
    true_labels = labels.flatten() if len(labels.shape) > 1 else labels
    
    # Ensure both are integers
    pred_classes = pred_classes.astype(int)
    true_labels = true_labels.astype(int)
    
    # Compute metrics
    try:
        accuracy = accuracy_score(true_labels, pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_classes, average='weighted', zero_division=0
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, pred_classes)
        
        # For binary classification, calculate specificity
        if len(np.unique(true_labels)) == 2:
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            specificity = 0
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'true_positives': tp if len(np.unique(true_labels)) == 2 else 0,
            'true_negatives': tn if len(np.unique(true_labels)) == 2 else 0,
            'false_positives': fp if len(np.unique(true_labels)) == 2 else 0,
            'false_negatives': fn if len(np.unique(true_labels)) == 2 else 0
        }
    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        # Return default values to prevent training from failing
        return {
            'accuracy': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 0.0
        }


def compare_model_performance(distillation_results, baseline_results):
    """
    Compare and analyze performance between distillation and baseline models.
    
    Args:
        distillation_results: Dictionary with distillation model evaluation results
        baseline_results: Dictionary with baseline model evaluation results
        
    Returns:
        Tuple of (DataFrame with comparison, Dictionary with improvement metrics)
    """
    # Extract key metrics
    distillation_accuracy = distillation_results.get('eval_accuracy', 0)
    distillation_f1 = distillation_results.get('eval_f1', 0)
    distillation_precision = distillation_results.get('eval_precision', 0)
    distillation_recall = distillation_results.get('eval_recall', 0)
    
    baseline_accuracy = baseline_results.get('eval_accuracy', 0)
    baseline_f1 = baseline_results.get('eval_f1', 0)
    baseline_precision = baseline_results.get('eval_precision', 0)
    baseline_recall = baseline_results.get('eval_recall', 0)
    
    # Create comparison dataframe
    comparison_data = {
        'Method': ['Tri-Model Distillation', 'Baseline (No Distillation)'],
        'Accuracy': [distillation_accuracy, baseline_accuracy],
        'F1-Score': [distillation_f1, baseline_f1],
        'Precision': [distillation_precision, baseline_precision],
        'Recall': [distillation_recall, baseline_recall]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate improvements
    accuracy_improvement = ((distillation_accuracy - baseline_accuracy) / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0
    f1_improvement = ((distillation_f1 - baseline_f1) / baseline_f1) * 100 if baseline_f1 > 0 else 0
    precision_improvement = ((distillation_precision - baseline_precision) / baseline_precision) * 100 if baseline_precision > 0 else 0
    recall_improvement = ((distillation_recall - baseline_recall) / baseline_recall) * 100 if baseline_recall > 0 else 0
    
    improvements = {
        'accuracy_improvement': accuracy_improvement,
        'f1_improvement': f1_improvement,
        'precision_improvement': precision_improvement,
        'recall_improvement': recall_improvement
    }
    
    return comparison_df, improvements


def calculate_efficiency_performance_tradeoff(student_info, teacher_info, distillation_accuracy, teacher_accuracy=0.88):
    """
    Calculate efficiency vs performance trade-off metrics.
    
    Args:
        student_info: Dictionary with student model information
        teacher_info: Dictionary with teacher model information
        distillation_accuracy: Accuracy achieved by distilled student model
        teacher_accuracy: Accuracy achieved by teacher model (default: 0.88)
        
    Returns:
        Dictionary with trade-off metrics
    """
    efficiency_ratio = student_info['total_params'] / teacher_info['total_params']
    performance_ratio = distillation_accuracy / teacher_accuracy
    efficiency_score = performance_ratio / efficiency_ratio
    
    return {
        'efficiency_ratio': efficiency_ratio,
        'performance_ratio': performance_ratio,
        'efficiency_score': efficiency_score,
        'param_reduction_percent': (1 - efficiency_ratio) * 100,
        'performance_retention_percent': performance_ratio * 100
    }


def print_performance_summary(comparison_df, improvements, tradeoff_metrics=None):
    """
    Print a comprehensive performance summary.
    
    Args:
        comparison_df: DataFrame with model comparison
        improvements: Dictionary with improvement metrics
        tradeoff_metrics: Optional dictionary with trade-off metrics
    """
    print("\nPerformance Comparison:")
    print("=" * 60)
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    print(f"\nImprovement with Tri-Model Distillation:")
    print(f"Accuracy: {improvements['accuracy_improvement']:+.2f}%")
    print(f"F1-Score: {improvements['f1_improvement']:+.2f}%")
    print(f"Precision: {improvements['precision_improvement']:+.2f}%")
    print(f"Recall: {improvements['recall_improvement']:+.2f}%")
    
    if tradeoff_metrics:
        print(f"\n🎯 EFFICIENCY vs PERFORMANCE TRADE-OFF:")
        print(f"   • Model Size Ratio: {tradeoff_metrics['efficiency_ratio']:.3f} ({tradeoff_metrics['param_reduction_percent']:.1f}% reduction)")
        print(f"   • Performance Ratio: {tradeoff_metrics['performance_ratio']:.3f} ({tradeoff_metrics['performance_retention_percent']:.1f}% of teacher performance)")
        print(f"   • Efficiency Score: {tradeoff_metrics['efficiency_score']:.3f} (higher is better)")


def save_evaluation_results(output_dir, test_results, comparison_df=None, improvements=None, tradeoff_metrics=None):
    """
    Save evaluation results to files.
    
    Args:
        output_dir: Directory to save results
        test_results: Dictionary with test results
        comparison_df: Optional DataFrame with model comparison
        improvements: Optional dictionary with improvement metrics
        tradeoff_metrics: Optional dictionary with trade-off metrics
    """
    import os
    import json
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save test results
    with open(os.path.join(output_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    
    # Save comparison if available
    if comparison_df is not None:
        comparison_df.to_csv(os.path.join(output_dir, "performance_comparison.csv"), index=False)
    
    # Save improvements and trade-off metrics if available
    if improvements is not None or tradeoff_metrics is not None:
        metrics = {}
        if improvements is not None:
            metrics.update(improvements)
        if tradeoff_metrics is not None:
            metrics.update(tradeoff_metrics)
            
        with open(os.path.join(output_dir, "performance_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
            
    logger.info(f"Evaluation results saved to {output_dir}")