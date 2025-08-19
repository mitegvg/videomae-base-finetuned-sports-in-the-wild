"""
Data Processing Utilities for Tri-Model Distillation Framework

This module provides functions for loading, processing, and managing data
for the tri-model distillation framework.
"""

import os
import torch
import numpy as np
import pandas as pd
import gc
from typing import Dict, List, Optional, Any, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def process_dataset_with_memory_optimization(dataset, processor, max_samples=None, batch_size=10):
    """
    Process video dataset with memory optimization.
    
    Args:
        dataset: Raw dataset with video paths and labels
        processor: VideoMAE image processor
        max_samples: Maximum number of samples to process (None for all)
        batch_size: Batch size for processing
        
    Returns:
        List of processed samples
    """
    processed_dataset = []
    
    # Determine how many samples to process
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    logger.info(f"Processing {num_samples} samples with batch size {batch_size}")
    
    # Process in batches
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch = dataset[i:end_idx]
        
        # Process batch
        batch_processed = []
        for sample in batch:
            try:
                # Get pixel values
                pixel_values = sample['pixel_values'] if 'pixel_values' in sample else processor(
                    list(sample['frames']),
                    return_tensors="pt"
                ).pixel_values.squeeze(0)
                
                # Create processed sample
                processed_sample = {
                    'pixel_values': pixel_values.cpu(),  # Keep on CPU to save GPU memory
                    'labels': sample['label'] if 'label' in sample else sample['labels']
                }
                
                batch_processed.append(processed_sample)
                
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
        
        # Add processed batch to dataset
        processed_dataset.extend(batch_processed)
        
        # Memory cleanup
        if i % (batch_size * 5) == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Processed {len(processed_dataset)}/{num_samples} samples")
    
    logger.info(f"Dataset processing complete. Total samples: {len(processed_dataset)}")
    return processed_dataset


def create_binary_label_mapping():
    """
    Create binary label mapping for datasets with A/B/G style labels.
    
    Returns:
        Tuple of (label2id, id2label) dictionaries
    """
    # Binary mapping: A=0 (safe), B1-B6,G=1 (unsafe)
    label2id = {
        'A': 0,   # Class 0
        'B1': 1,  # Class 1
        'B2': 1,  # Class 1
        'B4': 1,  # Class 1
        'B5': 1,  # Class 1
        'B6': 1,  # Class 1
        'G': 1    # Class 1
    }
    
    id2label = {
        0: 'safe',
        1: 'unsafe'
    }
    
    return label2id, id2label


def prepare_dataloaders(processed_train_dataset, processed_val_dataset, batch_size=4, num_workers=2):
    """
    Prepare data loaders for training and validation.
    
    Args:
        processed_train_dataset: Processed training dataset
        processed_val_dataset: Processed validation dataset
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    from torch.utils.data import DataLoader
    
    # Create data loaders
    train_dataloader = DataLoader(
        processed_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        processed_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader