"""
Visualization Utilities for Tri-Model Distillation Framework

This module provides functions for visualizing model architectures, feature alignments,
and performance comparisons.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from typing import Dict, List, Optional, Any


def visualize_model_comparison(teacher_info, student_info, assistant_info=None):
    """
    Create a comprehensive visualization comparing model architectures.
    
    Args:
        teacher_info: Dictionary with teacher model information
        student_info: Dictionary with student model information
        assistant_info: Optional dictionary with assistant model information
    
    Returns:
        Matplotlib figure with comparison visualizations
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Teacher vs Tiny Student Model Comparison', fontsize=16, fontweight='bold')
    
    # 1. Parameter Count Comparison
    ax1 = axes[0, 0]
    models = ['Teacher\n(VideoMAE)', 'Student\n(Tiny VideoMAE)']
    params = [teacher_info['total_params'], student_info['total_params']]
    colors = ['#ff7f0e', '#2ca02c']
    
    bars = ax1.bar(models, params, color=colors, alpha=0.7)
    ax1.set_title('Total Parameters')
    ax1.set_ylabel('Parameter Count')
    
    # Add value labels
    for bar, param in zip(bars, params):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params) * 0.02,
                f'{param/1000000:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    # 2. Architecture Comparison
    ax2 = axes[0, 1]
    metrics = ['Layers', 'Hidden\nSize', 'Attention\nHeads']
    teacher_values = [teacher_info.get('num_layers', 12), teacher_info.get('hidden_size', 768), teacher_info.get('attention_heads', 12)]
    student_values = [student_info.get('num_layers', 4), student_info.get('hidden_size', 384), student_info.get('attention_heads', 6)]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax2.bar(x - width/2, teacher_values, width, label='Teacher', color=colors[0], alpha=0.7)
    ax2.bar(x + width/2, student_values, width, label='Student', color=colors[1], alpha=0.7)
    
    ax2.set_title('Architecture Details')
    ax2.set_ylabel('Count/Size')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    
    # 3. Memory Usage
    ax3 = axes[0, 2]
    memory = [teacher_info['memory_mb'], student_info['memory_mb']]
    bars = ax3.bar(models, memory, color=colors, alpha=0.7)
    ax3.set_title('Memory Usage')
    ax3.set_ylabel('Memory (MB)')
    
    for bar, mem in zip(bars, memory):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(memory) * 0.02,
                f'{mem:.1f}MB', ha='center', va='bottom', fontweight='bold')
    
    # 4. Efficiency Pie Chart
    ax4 = axes[1, 0]
    efficiency_labels = ['Student\nModel', 'Reduction']
    efficiency_values = [student_info['total_params'], teacher_info['total_params'] - student_info['total_params']]
    efficiency_colors = ['#2ca02c', '#ffcccc']
    
    wedges, texts, autotexts = ax4.pie(efficiency_values, labels=efficiency_labels, colors=efficiency_colors, 
                                      autopct='%1.1f%%', startangle=90)
    ax4.set_title('Parameter Efficiency')
    
    # 5. Speed Comparison (Theoretical)
    ax5 = axes[1, 1]
    operations = ['Forward\nPass', 'Attention\nComputation', 'Overall\nInference']
    teacher_speed = [1.0, 1.0, 1.0]  # Normalized to 1.0
    student_speed = [4.0, 3.0, 3.5]  # Estimated speed improvements
    
    x = np.arange(len(operations))
    ax5.bar(x - width/2, teacher_speed, width, label='Teacher (baseline)', color=colors[0], alpha=0.7)
    ax5.bar(x + width/2, student_speed, width, label='Student (speedup)', color=colors[1], alpha=0.7)
    
    ax5.set_title('Inference Speed (Theoretical)')
    ax5.set_ylabel('Relative Speed')
    ax5.set_xticks(x)
    ax5.set_xticklabels(operations)
    ax5.legend()
    
    # 6. Performance Expectation
    ax6 = axes[1, 2]
    scenarios = ['Baseline\n(no training)', 'Direct\nFine-tuning', 'Tri-Model\nDistillation']
    expected_performance = [60, 75, 85]  # Expected accuracy percentages
    performance_colors = ['#ff9999', '#ffcc99', '#99ff99']
    
    bars = ax6.bar(scenarios, expected_performance, color=performance_colors, alpha=0.8)
    ax6.set_title('Expected Performance')
    ax6.set_ylabel('Accuracy (%)')
    ax6.set_ylim(0, 100)
    
    # Add target line for teacher performance
    ax6.axhline(y=88, color='red', linestyle='--', alpha=0.7, label='Teacher Target')
    ax6.legend()
    
    for bar, perf in zip(bars, expected_performance):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{perf}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig


def visualize_feature_alignment(student_feat, teacher_feat, aligned_teacher):
    """
    Visualize feature alignment between teacher and student models.
    
    Args:
        student_feat: Student model features
        teacher_feat: Original teacher model features
        aligned_teacher: Aligned teacher features
        
    Returns:
        Matplotlib figure with feature alignment visualizations
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Feature Alignment Process Visualization', fontsize=14, fontweight='bold')
    
    # Convert tensors to numpy for visualization (take first sample, mean over sequence)
    teacher_np = teacher_feat[0].mean(dim=0).cpu().numpy()
    student_np = student_feat[0].mean(dim=0).cpu().numpy()
    aligned_teacher_np = aligned_teacher[0].mean(dim=0).cpu().numpy()
    
    # Plot 1: Teacher features before alignment
    axes[0].hist(teacher_np, bins=50, alpha=0.7, color='orange', density=True)
    axes[0].set_title('Teacher Features\n(Before Alignment)')
    axes[0].set_xlabel('Feature Value')
    axes[0].set_ylabel('Density')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Teacher features after alignment
    axes[1].hist(aligned_teacher_np, bins=50, alpha=0.7, color='lightblue', density=True)
    axes[1].set_title('Teacher Features\n(After Alignment)')
    axes[1].set_xlabel('Feature Value')
    axes[1].set_ylabel('Density')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Student features (target)
    axes[2].hist(student_np, bins=50, alpha=0.7, color='green', density=True)
    axes[2].set_title('Student Features\n(Target)')
    axes[2].set_xlabel('Feature Value')
    axes[2].set_ylabel('Density')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_alignment_strategy():
    """
    Create a diagram illustrating the feature alignment strategy.
    
    Returns:
        Matplotlib figure with alignment strategy diagram
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title('Feature Alignment Strategy', fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    
    # Draw alignment flow
    # Teacher box
    teacher_box = FancyBboxPatch((0.5, 7), 2, 1.5, boxstyle="round,pad=0.1", 
                                facecolor=colors[0], alpha=0.7, edgecolor='black')
    ax.add_patch(teacher_box)
    ax.text(1.5, 7.75, 'Teacher\nFeatures', ha='center', va='center', fontweight='bold')
    
    # Assistant box
    assistant_box = FancyBboxPatch((0.5, 4.5), 2, 1.5, boxstyle="round,pad=0.1",
                                  facecolor=colors[2], alpha=0.7, edgecolor='black')
    ax.add_patch(assistant_box)
    ax.text(1.5, 5.25, 'Assistant\nFeatures', ha='center', va='center', fontweight='bold')
    
    # Alignment box
    align_box = FancyBboxPatch((4, 5.5), 3, 2, boxstyle="round,pad=0.1",
                              facecolor='lightblue', alpha=0.7, edgecolor='black')
    ax.add_patch(align_box)
    ax.text(5.5, 6.5, 'Feature\nAlignment\n(Linear Proj +\nInterpolation)', 
             ha='center', va='center', fontweight='bold')
    
    # Student box
    student_box = FancyBboxPatch((8, 5.5), 2, 2, boxstyle="round,pad=0.1",
                                facecolor=colors[1], alpha=0.7, edgecolor='black')
    ax.add_patch(student_box)
    ax.text(9, 6.5, 'Student\nModel\n(Target)', ha='center', va='center', fontweight='bold')
    
    # Draw arrows
    ax.annotate('', xy=(4, 6.8), xytext=(2.5, 7.5), 
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(4, 6.2), xytext=(2.5, 5.5), 
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(8, 6.5), xytext=(7, 6.5), 
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    plt.tight_layout()
    return fig


def visualize_performance_comparison(comparison_df):
    """
    Visualize performance comparison between different models.
    
    Args:
        comparison_df: DataFrame with performance metrics
        
    Returns:
        Matplotlib figure with performance comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Tri-Model Distillation vs Baseline Performance', fontsize=16)
    
    metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    colors = ['skyblue', 'lightcoral']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        values = comparison_df[metric].values
        bars = ax.bar(comparison_df['Method'], values, color=colors)
        ax.set_title(metric)
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig