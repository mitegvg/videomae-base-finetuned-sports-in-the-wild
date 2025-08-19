"""
Model Analysis Utilities for Tri-Model Distillation Framework

This module provides functions for analyzing model architectures, parameters,
and performance metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any


def analyze_model_architecture(model, model_name):
    """
    Analyze and return detailed architecture information.
    
    Args:
        model: The model to analyze
        model_name: A descriptive name for the model
        
    Returns:
        Dictionary containing model architecture details
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get configuration if available
    config = getattr(model, 'config', None)
    
    if config:
        info = {
            'name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_layers': getattr(config, 'num_hidden_layers', 'N/A'),
            'hidden_size': getattr(config, 'hidden_size', 'N/A'),
            'attention_heads': getattr(config, 'num_attention_heads', 'N/A'), 
            'intermediate_size': getattr(config, 'intermediate_size', 'N/A'),
            'num_labels': getattr(config, 'num_labels', 'N/A'),
            'memory_mb': total_params * 4 / (1024 * 1024),  # Approximate MB (4 bytes per param)
        }
    else:
        info = {
            'name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'memory_mb': total_params * 4 / (1024 * 1024),
        }
    
    return info


def calculate_efficiency_metrics(teacher_info, student_info):
    """
    Calculate efficiency metrics comparing teacher and student models.
    
    Args:
        teacher_info: Dictionary with teacher model information
        student_info: Dictionary with student model information
        
    Returns:
        Dictionary with efficiency metrics
    """
    param_reduction = (teacher_info['total_params'] - student_info['total_params']) / teacher_info['total_params'] * 100
    memory_reduction = (teacher_info['memory_mb'] - student_info['memory_mb']) / teacher_info['memory_mb'] * 100
    speed_increase = teacher_info.get('num_layers', 12) / student_info.get('num_layers', 4)
    
    return {
        'param_reduction': param_reduction,
        'memory_reduction': memory_reduction,
        'speed_increase': speed_increase,
        'efficiency_ratio': student_info['total_params'] / teacher_info['total_params']
    }


def print_model_comparison(teacher_info, student_info, assistant_info=None):
    """
    Print a detailed comparison of model architectures.
    
    Args:
        teacher_info: Dictionary with teacher model information
        student_info: Dictionary with student model information
        assistant_info: Optional dictionary with assistant model information
    """
    print("🔍 ARCHITECTURE COMPARISON:")
    print("-" * 60)
    
    models_info = [teacher_info, student_info]
    if assistant_info:
        models_info.append(assistant_info)
    
    for info in models_info:
        print(f"\n📱 {info['name']}:")
        print(f"   Total Parameters: {info['total_params']:,}")
        print(f"   Trainable Parameters: {info['trainable_params']:,}")
        print(f"   Memory Usage: ~{info['memory_mb']:.1f} MB")
        
        if 'num_layers' in info and info['num_layers'] != 'N/A':
            print(f"   Layers: {info['num_layers']}")
            print(f"   Hidden Size: {info['hidden_size']}")
            print(f"   Attention Heads: {info['attention_heads']}")
            print(f"   Intermediate Size: {info['intermediate_size']:,}")
    
    # Calculate efficiency gains
    metrics = calculate_efficiency_metrics(teacher_info, student_info)
    
    print(f"\n🎯 EFFICIENCY GAINS:")
    print(f"   Parameter Reduction: {metrics['param_reduction']:.1f}%")
    print(f"   Memory Reduction: {metrics['memory_reduction']:.1f}%") 
    print(f"   Expected Speed Increase: ~{metrics['speed_increase']:.1f}x")