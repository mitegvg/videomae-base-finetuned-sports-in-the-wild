"""
Testing Utilities for Tri-Model Distillation Framework

This module provides functions for testing and validating the functionality
of the tri-model distillation framework, especially feature alignment.
"""

import torch
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def test_attention_alignment(loss_fn, device):
    """
    Test attention alignment functionality with various test scenarios.
    
    Args:
        loss_fn: TriModelDistillationLoss instance
        device: Torch device to run tests on
        
    Returns:
        Dictionary with test results
    """
    print("🧪 Testing Attention Alignment Functionality")
    print("="*50)
    
    # Test scenarios for attention alignment
    test_scenarios = [
        {
            'name': 'Same dimensions',
            'student_shape': (2, 6, 197, 197),
            'teacher_shape': (2, 6, 197, 197)
        },
        {
            'name': 'Different heads (6 vs 12)',
            'student_shape': (2, 6, 197, 197),
            'teacher_shape': (2, 12, 197, 197)
        },
        {
            'name': 'Different sequence length',
            'student_shape': (2, 6, 100, 100),
            'teacher_shape': (2, 6, 197, 197)
        },
        {
            'name': 'Both different (heads and sequence)',
            'student_shape': (2, 6, 100, 100),
            'teacher_shape': (2, 12, 197, 197)
        },
        {
            'name': 'Real model dimensions (tiny student vs teacher)',
            'student_shape': (4, 6, 1568, 1568),  # Real tiny student
            'teacher_shape': (4, 12, 1568, 1568)  # Real teacher
        }
    ]
    
    print(f"Running {len(test_scenarios)} test scenarios...")
    print()
    
    results = {}
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"Test {i}: {scenario['name']}")
        
        # Create test tensors
        student_attn = torch.randn(scenario['student_shape'], device=device)
        teacher_attn = torch.randn(scenario['teacher_shape'], device=device)
        
        try:
            # Test the alignment function
            aligned_teacher = loss_fn._align_attention_dimensions(student_attn, teacher_attn)
            
            # Verify dimensions match
            if aligned_teacher.shape == student_attn.shape:
                print(f"  [OK] SUCCESS: {scenario['teacher_shape']} -> {aligned_teacher.shape}")
                
                # Test that we can compute MSE loss without error
                mse_loss = loss_fn.mse_loss(student_attn, aligned_teacher)
                print(f"  [OK] MSE Loss computed: {mse_loss.item():.6f}")
                
                results[scenario['name']] = {
                    'success': True,
                    'mse_loss': mse_loss.item(),
                    'original_shape': scenario['teacher_shape'],
                    'aligned_shape': tuple(aligned_teacher.shape)
                }
            else:
                print(f"  [ERROR] FAILED: Shape mismatch! Expected {student_attn.shape}, got {aligned_teacher.shape}")
                results[scenario['name']] = {
                    'success': False,
                    'error': 'Shape mismatch',
                    'expected': tuple(student_attn.shape),
                    'got': tuple(aligned_teacher.shape)
                }
                
        except Exception as e:
            print(f"  [ERROR] ERROR: {str(e)}")
            results[scenario['name']] = {
                'success': False,
                'error': str(e)
            }
        
        print()
    
    print("🎯 Attention alignment testing complete!")
    
    # Test full attention distillation loss computation
    print("🔧 Testing Full Attention Distillation Loss...")
    print("-" * 30)
    
    try:
        # Create sample attention lists (simulating multi-layer)
        student_attentions = [torch.randn(4, 6, 1568, 1568, device=device) for _ in range(4)]
        teacher_attentions = [torch.randn(4, 12, 1568, 1568, device=device) for _ in range(12)]
        
        # Compute attention distillation loss
        attention_loss = loss_fn._compute_attention_distillation_loss(
            student_attentions, teacher_attentions
        )
        
        print(f"[OK] Attention distillation loss computed successfully: {attention_loss.item():.6f}")
        results['full_attention_loss'] = {
            'success': True,
            'loss_value': attention_loss.item()
        }
        
    except Exception as e:
        print(f"[ERROR] Error in attention distillation: {str(e)}")
        import traceback
        traceback.print_exc()
        results['full_attention_loss'] = {
            'success': False,
            'error': str(e)
        }
    
    return results


def test_feature_alignment(loss_fn, device):
    """
    Test feature alignment functionality with various test scenarios.
    
    Args:
        loss_fn: TriModelDistillationLoss instance
        device: Torch device to run tests on
        
    Returns:
        Dictionary with test results
    """
    print("[INFO] FEATURE ALIGNMENT FUNCTION DEMONSTRATION")
    print("=" * 60)
    
    results = {}
    
    # Test different alignment scenarios
    test_scenarios = [
        ("Same dimensions", torch.randn(1, 197, 768), torch.randn(1, 197, 768)),
        ("Different feature dims", torch.randn(1, 197, 512), torch.randn(1, 197, 768)),
        ("Different sequence length", torch.randn(1, 100, 768), torch.randn(1, 197, 768)),
        ("Both different", torch.randn(1, 150, 512), torch.randn(1, 197, 768)),
        ("Real model dimensions", torch.randn(1, 1568, 384), torch.randn(1, 1568, 768))
    ]
    
    print(f"🧪 TESTING DIFFERENT ALIGNMENT SCENARIOS:")
    
    for scenario_name, source_tensor, target_tensor in test_scenarios:
        source_tensor = source_tensor.to(device)
        target_tensor = target_tensor.to(device)
        
        try:
            aligned_tensor = loss_fn._align_feature_dimensions(source_tensor, target_tensor)
            
            print(f"   {scenario_name}:")
            print(f"     Source: {source_tensor.shape} → Aligned: {aligned_tensor.shape} → Target: {target_tensor.shape}")
            print(f"     Success: {aligned_tensor.shape == target_tensor.shape}")
            
            results[scenario_name] = {
                'success': aligned_tensor.shape == target_tensor.shape,
                'source_shape': tuple(source_tensor.shape),
                'aligned_shape': tuple(aligned_tensor.shape),
                'target_shape': tuple(target_tensor.shape)
            }
            
            # Calculate similarity if successful
            if aligned_tensor.shape == target_tensor.shape:
                similarity = torch.cosine_similarity(
                    aligned_tensor.flatten(1), 
                    target_tensor.flatten(1), 
                    dim=1
                ).mean().item()
                
                results[scenario_name]['similarity'] = similarity
                
        except Exception as e:
            print(f"   {scenario_name}: [ERROR] ERROR - {str(e)}")
            results[scenario_name] = {
                'success': False,
                'error': str(e)
            }
    
    return results