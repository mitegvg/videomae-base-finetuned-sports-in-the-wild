"""
Inference Utilities for Tri-Model Distillation Framework

This module provides functions for running inference with trained models
from the tri-model distillation framework.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


def predict_video_class(video_path, model, processor, device, classification_type="binary", 
                        num_classes=2, class_names=None, id2label=None):
    """
    Predict class for a video using a trained VideoMAE model.
    
    Args:
        video_path: Path to the video file
        model: Trained VideoMAE model
        processor: VideoMAE image processor
        device: Torch device to run inference on
        classification_type: Type of classification ("binary", "multiclass", "multilabel")
        num_classes: Number of classes
        class_names: Optional list of class names
        id2label: Optional mapping from class IDs to labels
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Process video
        from decord import VideoReader, cpu
        import numpy as np
        
        # Load video
        videoreader = VideoReader(video_path, ctx=cpu(0))
        video_length = len(videoreader)
        
        # Sample frames uniformly (16 frames)
        sample_indices = np.linspace(0, video_length - 1, 16, dtype=int)
        frames = videoreader.get_batch(sample_indices).asnumpy()
        
        # Preprocess frames
        pixel_values = processor(
            list(frames),
            return_tensors="pt",
            sampling_rate=1
        ).pixel_values
        
        # Move to device
        pixel_values = pixel_values.to(device)
        
        # Generate class names if not provided
        if class_names is None:
            if id2label is not None:
                class_names = [id2label[i] for i in range(num_classes)]
            else:
                class_names = [f'Class_{i}' for i in range(num_classes)]
        
        # Run inference
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
            if classification_type == "multilabel":
                # For multilabel, use sigmoid activation
                probabilities = torch.sigmoid(logits)
                # Threshold for multilabel predictions (can be made configurable)
                threshold = 0.5
                predictions = (probabilities > threshold).float()
                predicted_classes = [class_names[i] for i in range(num_classes) if predictions[0][i] == 1]
                
                # Get top confidence scores
                top_confidences = {class_names[i]: probabilities[0][i].item() for i in range(num_classes)}
                
                return {
                    'classification_type': classification_type,
                    'predicted_classes': predicted_classes,
                    'probabilities': top_confidences,
                    'threshold': threshold
                }
            else:
                # For binary and multiclass, use softmax
                probabilities = F.softmax(logits, dim=-1)
                predicted_class_idx = torch.argmax(logits, dim=-1).item()
                predicted_class = class_names[predicted_class_idx]
                confidence = probabilities[0][predicted_class_idx].item()
                
                # Get all class probabilities
                class_probabilities = {class_names[i]: probabilities[0][i].item() for i in range(num_classes)}
                
                return {
                    'classification_type': classification_type,
                    'predicted_class': predicted_class,
                    'predicted_class_idx': predicted_class_idx,
                    'confidence': confidence,
                    'probabilities': class_probabilities
                }
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        return {'error': str(e)}


def batch_inference(video_paths, model, processor, device, batch_size=4, 
                   classification_type="binary", num_classes=2, class_names=None, id2label=None):
    """
    Run inference on a batch of videos.
    
    Args:
        video_paths: List of paths to video files
        model: Trained VideoMAE model
        processor: VideoMAE image processor
        device: Torch device to run inference on
        batch_size: Batch size for inference
        classification_type: Type of classification ("binary", "multiclass", "multilabel")
        num_classes: Number of classes
        class_names: Optional list of class names
        id2label: Optional mapping from class IDs to labels
        
    Returns:
        List of dictionaries with prediction results
    """
    results = []
    
    # Process videos in batches
    for i in range(0, len(video_paths), batch_size):
        batch_paths = video_paths[i:i+batch_size]
        batch_results = []
        
        for video_path in batch_paths:
            result = predict_video_class(
                video_path, model, processor, device, 
                classification_type, num_classes, class_names, id2label
            )
            batch_results.append(result)
        
        results.extend(batch_results)
    
    return results


def create_demo_inference(model, processor, device, sample_data, 
                         classification_type="binary", num_classes=2, class_names=None, id2label=None):
    """
    Create a demo inference using a sample from processed dataset.
    
    Args:
        model: Trained VideoMAE model
        processor: VideoMAE image processor
        device: Torch device to run inference on
        sample_data: Sample data from processed dataset
        classification_type: Type of classification ("binary", "multiclass", "multilabel")
        num_classes: Number of classes
        class_names: Optional list of class names
        id2label: Optional mapping from class IDs to labels
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Generate class names if not provided
        if class_names is None:
            if id2label is not None:
                class_names = [id2label[i] for i in range(num_classes)]
            else:
                class_names = [f'Class_{i}' for i in range(num_classes)]
        
        with torch.no_grad():
            inputs = {
                'pixel_values': sample_data['pixel_values'].unsqueeze(0).to(device)
            }
            
            outputs = model(**inputs)
            logits = outputs.logits
            
            if classification_type == "multilabel":
                # For multilabel, use sigmoid activation
                probabilities = torch.sigmoid(logits)
                threshold = 0.5
                predictions = (probabilities > threshold).float()
                predicted_classes = [class_names[i] for i in range(num_classes) if predictions[0][i] == 1]
                
                # Get top confidence scores
                top_confidences = {class_names[i]: probabilities[0][i].item() for i in range(num_classes)}
                
                return {
                    'classification_type': classification_type,
                    'predicted_classes': predicted_classes,
                    'probabilities': top_confidences,
                    'threshold': threshold
                }
            else:
                # For binary and multiclass, use softmax
                probabilities = F.softmax(logits, dim=-1)
                predicted_class_idx = torch.argmax(logits, dim=-1).item()
                predicted_class = class_names[predicted_class_idx]
                confidence = probabilities[0][predicted_class_idx].item()
                
                # Get all class probabilities
                class_probabilities = {class_names[i]: probabilities[0][i].item() for i in range(num_classes)}
                
                return {
                    'classification_type': classification_type,
                    'predicted_class': predicted_class,
                    'predicted_class_idx': predicted_class_idx,
                    'confidence': confidence,
                    'probabilities': class_probabilities
                }
    except Exception as e:
        logger.error(f"Error during demo inference: {str(e)}")
        return {'error': str(e)}