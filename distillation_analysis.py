#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive Distillation Analysis Script

This script analyzes the tri-model distillation framework from three key perspectives:
1. Decision boundaries and logit structures
2. Semantic alignment between teacher/student knowledge
3. Deployment metrics for edge devices

"""

import os
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm 
from pathlib import Path
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, pipeline
import cv2
import warnings
import psutil
import gc
from scipy import stats

# Try to import thop for FLOPs calculation
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    print("Warning: thop not available. FLOPs calculation will be skipped.")
    print("Install with: pip install thop")
    THOP_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
MODELS = {
    "teacher": "mitegvg/videomae-base-kinetics-binary-finetuned-xd-violence",
    "assistant": "mitegvg/videomae-base-ssv2-binary-finetuned-xd-violence", 
    "student": "mitegvg/videomae-tiny-binary-finetuned-xd-violence",
    "distilled": "tri_model_distillation_production_20250731_185621/checkpoint-1050"
}

TEST_CSV = "processed_dataset/test.csv"
RESULTS_DIR = "distillation_analysis_results"
SAMPLE_SIZE = 100  # Number of videos to analyze for detailed metrics

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DistillationAnalyzer:
    """Comprehensive analyzer for distillation framework"""
    
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.results = {}
        
    def load_models(self):
        """Load all models for analysis"""
        print("Loading models for distillation analysis...")
        
        for model_name, model_path in MODELS.items():
            print(f"\nLoading {model_name} model: {model_path}")
            try:
                # Load model and processor
                if model_name == "distilled" and os.path.exists(model_path):
                    # Local distilled model
                    model = VideoMAEForVideoClassification.from_pretrained(model_path)
                    processor = VideoMAEImageProcessor.from_pretrained(model_path)
                else:
                    # HuggingFace models
                    model = VideoMAEForVideoClassification.from_pretrained(model_path)
                    processor = VideoMAEImageProcessor.from_pretrained(model_path)
                
                model.to(device)
                model.eval()
                
                self.models[model_name] = model
                self.processors[model_name] = processor
                
                print(f"✓ Successfully loaded {model_name} model")
                
            except Exception as e:
                print(f"✗ Failed to load {model_name} model: {e}")
                continue
    
    def get_model_size(self, model):
        """Calculate model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def calculate_flops(self, model, input_shape=(1, 16, 3, 224, 224)):
        """Calculate FLOPs for model"""
        if not THOP_AVAILABLE:
            return 0, "N/A", 0, "N/A"
            
        try:
            dummy_input = torch.randn(input_shape).to(device)
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
            return flops, flops_formatted, params, params_formatted
        except Exception as e:
            print(f"FLOP calculation failed: {e}")
            return 0, "N/A", 0, "N/A"
    
    def measure_inference_latency(self, model, processor, num_runs=50):
        """Measure inference latency"""
        print(f"Measuring inference latency ({num_runs} runs)...")
        
        # Create dummy video input
        dummy_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
        
        latencies = []
        
        # Warmup runs
        for _ in range(5):
            with torch.no_grad():
                inputs = processor(dummy_frames, return_tensors="pt", do_rescale=False)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                _ = model(**inputs)
        
        # Actual timing runs
        for _ in tqdm(range(num_runs), desc="Latency measurement"):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                inputs = processor(dummy_frames, return_tensors="pt", do_rescale=False)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            "mean_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
            "median_latency_ms": np.median(latencies)
        }
    
    def extract_features_and_logits(self, model, processor, video_paths, labels):
        """Extract features and logits from model"""
        features = []
        logits = []
        predictions = []
        all_logits = []  # Store full logits for Top-k accuracy calculation
        
        print(f"Extracting features and logits from {len(video_paths)} videos...")
        
        for video_path, label in tqdm(zip(video_paths, labels), total=len(video_paths)):
            try:
                # Read and process video
                frames = self.read_video(video_path)
                if frames is None:
                    continue
                
                inputs = processor(frames, return_tensors="pt", do_rescale=False)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    
                    # Extract logits
                    logit = outputs.logits.cpu().numpy().flatten()
                    logits.append(logit)
                    
                    # Store full logits (not flattened) for Top-k accuracy
                    all_logits.append(outputs.logits.cpu().numpy())
                    
                    # Extract last hidden state as features
                    hidden_states = outputs.hidden_states[-1]  # Last layer
                    feature = hidden_states.mean(dim=1).cpu().numpy().flatten()  # Global average pooling
                    features.append(feature)
                    
                    # Get prediction
                    pred = torch.argmax(outputs.logits, dim=1).item()
                    predictions.append(pred)
                    
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
        
        # Stack all_logits if not empty
        stacked_all_logits = np.vstack(all_logits) if all_logits else np.array([])
        
        return np.array(features), np.array(logits), np.array(predictions), stacked_all_logits
    
    def calculate_topk_accuracy(self, logits, labels, k_values=[1, 5]):
        """Calculate Top-k accuracy for multiple k values
        
        Args:
            logits: Raw model output logits with shape [n_samples, n_classes]
            labels: Ground truth labels with shape [n_samples]
            k_values: List of k values for Top-k accuracy
            
        Returns:
            Dictionary with Top-k accuracy for each k
        """
        if len(logits) == 0 or len(labels) == 0:
            return {f"top{k}_accuracy": 0.0 for k in k_values}
        
        # Ensure labels are the right shape
        if len(labels) > len(logits):
            labels = labels[:len(logits)]
        
        # Get shape information
        n_samples, n_classes = logits.shape
        
        # Sort indices in descending order of logits
        topk_indices = np.argsort(-logits, axis=1)
        
        # Create result dictionary
        results = {}
        
        # Calculate accuracy for each k
        for k in k_values:
            # Limit k to number of classes
            effective_k = min(k, n_classes)
            
            # Get top-k predictions for each sample
            top_k_preds = topk_indices[:, :effective_k]
            
            # Check if true label is in top-k predictions for each sample
            correct = np.any(top_k_preds == labels.reshape(-1, 1), axis=1)
            
            # Calculate accuracy
            accuracy = np.mean(correct)
            
            # Store result
            results[f"top{k}_accuracy"] = float(accuracy)
        
        return results
    
    def read_video(self, video_path, num_frames=16):
        """Read video and return frames"""
        try:
            # Ensure path is correct
            if not os.path.exists(video_path) and 'processed_dataset' not in video_path:
                video_path = os.path.join('processed_dataset', video_path)
            
            if not os.path.exists(video_path):
                return None
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < num_frames:
                cap.release()
                return None
            
            # Sample frames uniformly
            frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
            frames = []
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB and resize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            
            cap.release()
            
            if len(frames) != num_frames:
                return None
            
            return frames
            
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            return None
    
    def analyze_decision_boundaries(self, sample_data):
        """Analyze decision boundaries and logit structures (multiclass-ready)"""
        print("\n" + "="*60)
        print("ANALYZING DECISION BOUNDARIES AND LOGIT STRUCTURES")
        print("="*60)
        
        boundary_analysis = {}
        
        for model_name in self.models.keys():
            print(f"\nAnalyzing {model_name} model...")
            
            model = self.models[model_name]
            processor = self.processors[model_name]
            
            # Extract features, logits, and predictions
            features, logits, predictions, all_logits = self.extract_features_and_logits(
                model, processor, sample_data['video_paths'], sample_data['labels']
            )
            
            if len(logits) == 0:
                print(f"No valid predictions for {model_name}")
                continue
            
            # Subset labels to match processed logits count
            labels_aligned = sample_data['labels'][:len(logits)]

            # Calculate Top-k accuracy
            topk_accuracy = self.calculate_topk_accuracy(all_logits, labels_aligned, k_values=[1, 5])

            # Per-class statistics
            unique_classes = np.unique(labels_aligned)
            class_counts = {int(c): int(np.sum(labels_aligned == c)) for c in unique_classes}
            class_logit_stats = {}
            for c in unique_classes:
                cls_logits = logits[labels_aligned == c]
                class_logit_stats[int(c)] = {
                    'mean': np.mean(cls_logits, axis=0).tolist() if len(cls_logits) > 0 else None,
                    'std': np.std(cls_logits, axis=0).tolist() if len(cls_logits) > 0 else None,
                    'count': int(len(cls_logits)),
                }

            # Multiclass centroid separations
            separation = self.calculate_multiclass_logit_separation(logits, labels_aligned)

            boundary_analysis[model_name] = {
                'total_samples': int(len(logits)),
                'num_classes_present': int(len(unique_classes)),
                'class_counts': class_counts,
                'class_logit_stats': class_logit_stats,
                'decision_boundary_margin': self.calculate_decision_margin(logits, labels_aligned),
                'pairwise_logit_separation': separation,
                'confidence_distribution': self.analyze_confidence_distribution(logits),
                'accuracy': topk_accuracy
            }
        
        # Save boundary analysis
        self.save_results(boundary_analysis, "decision_boundary_analysis.json")
        
        # Create visualizations
        self.plot_decision_boundaries(boundary_analysis)
        
        return boundary_analysis
    
    def calculate_decision_margin(self, logits, labels):
        """Calculate decision boundary margin"""
        if logits.shape[1] < 2:
            # Ensure consistent keys even when only one class is present
            return {
                "mean_margin": 0.0,
                "std_margin": 0.0,
                "min_margin": 0.0,
                "max_margin": 0.0,
            }
        
        # Calculate margin (difference between top two logits)
        sorted_logits = np.sort(logits, axis=1)
        margins = sorted_logits[:, -1] - sorted_logits[:, -2]
        
        return {
            "mean_margin": float(np.mean(margins)),
            "std_margin": float(np.std(margins)),
            "min_margin": float(np.min(margins)),
            "max_margin": float(np.max(margins))
        }
    
    def calculate_multiclass_logit_separation(self, logits, labels):
        """Calculate average pairwise distance between class centroids in logit space."""
        if logits is None or len(logits) == 0:
            return {"mean_pairwise_distance": 0.0, "num_classes": 0}

        labels = np.asarray(labels)
        classes = np.unique(labels)
        if len(classes) < 2:
            return {"mean_pairwise_distance": 0.0, "num_classes": int(len(classes))}

        centroids = []
        for c in classes:
            cls_logits = logits[labels == c]
            if len(cls_logits) == 0:
                continue
            centroids.append(np.mean(cls_logits, axis=0))

        if len(centroids) < 2:
            return {"mean_pairwise_distance": 0.0, "num_classes": int(len(classes))}

        centroids = np.stack(centroids, axis=0)
        # Compute all pairwise distances
        from itertools import combinations
        dists = []
        for i, j in combinations(range(len(centroids)), 2):
            dists.append(np.linalg.norm(centroids[i] - centroids[j]))
        mean_dist = float(np.mean(dists)) if len(dists) > 0 else 0.0
        return {"mean_pairwise_distance": mean_dist, "num_classes": int(len(classes))}
    
    def analyze_confidence_distribution(self, logits):
        """Analyze confidence distribution"""
        probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
        max_probs = np.max(probabilities, axis=1)
        
        return {
            "mean_confidence": float(np.mean(max_probs)),
            "std_confidence": float(np.std(max_probs)),
            "low_confidence_ratio": float(np.mean(max_probs < 0.6)),
            "high_confidence_ratio": float(np.mean(max_probs > 0.9))
        }
    
    def analyze_semantic_alignment(self, sample_data):
        """Analyze semantic alignment between models"""
        print("\n" + "="*60)
        print("ANALYZING SEMANTIC ALIGNMENT")
        print("="*60)
        
        alignment_results = {}
        
        # Extract features from all models
        model_features = {}
        model_logits = {}
        
        for model_name in self.models.keys():
            print(f"Extracting features from {model_name}...")
            features, logits, _, _ = self.extract_features_and_logits(
                self.models[model_name], 
                self.processors[model_name],
                sample_data['video_paths'], 
                sample_data['labels']
            )
            model_features[model_name] = features
            model_logits[model_name] = logits
        
        # Calculate pairwise similarities
        for model1 in model_features.keys():
            for model2 in model_features.keys():
                if model1 >= model2:  # Avoid duplicate comparisons
                    continue
                
                pair_key = f"{model1}_vs_{model2}"
                print(f"Analyzing alignment: {pair_key}")
                
                # Feature similarity
                if len(model_features[model1]) > 0 and len(model_features[model2]) > 0:
                    feature_similarity = self.calculate_cosine_similarity(
                        model_features[model1], model_features[model2]
                    )
                else:
                    feature_similarity = {"mean": 0, "std": 0}
                
                # Logit similarity
                if len(model_logits[model1]) > 0 and len(model_logits[model2]) > 0:
                    logit_similarity = self.calculate_cosine_similarity(
                        model_logits[model1], model_logits[model2]
                    )
                else:
                    logit_similarity = {"mean": 0, "std": 0}
                
                alignment_results[pair_key] = {
                    "feature_similarity": feature_similarity,
                    "logit_similarity": logit_similarity,
                    "feature_correlation": self.calculate_feature_correlation(
                        model_features[model1], model_features[model2]
                    )
                }
        
        # Create t-SNE visualizations
        self.create_tsne_visualizations(model_features, sample_data['labels'], label_names=[sample_data['id_to_label'][i] for i in range(len(sample_data['id_to_label']))])
        
        # Save alignment results
        self.save_results(alignment_results, "semantic_alignment_analysis.json")
        
        return alignment_results
    
    def calculate_cosine_similarity(self, features1, features2):
        """Calculate cosine similarity between feature sets"""
        if len(features1) != len(features2) or len(features1) == 0:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        
        # Check if feature dimensions are compatible
        if features1.shape[1] != features2.shape[1]:
            print(f"Warning: Incompatible feature dimensions: {features1.shape[1]} vs {features2.shape[1]}")
            print("Using dimension reduction to compare features...")
            
            # Use PCA to reduce to common dimension
            # Target dimension must be <= min(n_samples, n_features) for both datasets
            n_samples = min(features1.shape[0], features2.shape[0])
            min_feature_dim = min(features1.shape[1], features2.shape[1])
            max_components = min(n_samples, min_feature_dim)
            target_dim = min(max_components - 1, 128)  # Leave some margin and cap at 128
            
            if target_dim < 1:
                print("Warning: Not enough data for PCA reduction, using direct comparison of first dimensions")
                # Fallback: just use the first min_feature_dim dimensions
                features1 = features1[:, :min_feature_dim]
                features2 = features2[:, :min_feature_dim]
            else:
                from sklearn.decomposition import PCA
                
                # Reduce features1
                pca1 = PCA(n_components=target_dim)
                features1_reduced = pca1.fit_transform(features1)
                
                # Reduce features2
                pca2 = PCA(n_components=target_dim)
                features2_reduced = pca2.fit_transform(features2)
                
                features1 = features1_reduced
                features2 = features2_reduced
                print(f"Reduced features to {target_dim} dimensions")
        
        similarities = []
        for f1, f2 in zip(features1, features2):
            sim = cosine_similarity([f1], [f2])[0, 0]
            similarities.append(sim)
        
        return {
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities))
        }
    
    def calculate_feature_correlation(self, features1, features2):
        """Calculate feature correlation"""
        if len(features1) != len(features2) or len(features1) == 0:
            return 0
        
        # Handle different feature dimensions
        if features1.shape[1] != features2.shape[1]:
            print(f"Warning: Different feature dimensions for correlation: {features1.shape[1]} vs {features2.shape[1]}")
            
            # Use PCA to reduce to common dimension for correlation analysis
            n_samples = min(features1.shape[0], features2.shape[0])
            min_feature_dim = min(features1.shape[1], features2.shape[1])
            max_components = min(n_samples, min_feature_dim)
            target_dim = min(max_components - 1, 64)  # Use fewer dimensions for correlation
            
            if target_dim < 1:
                print("Warning: Not enough data for PCA reduction in correlation, using direct comparison")
                # Fallback: just use the first min_feature_dim dimensions
                features1 = features1[:, :min_feature_dim]
                features2 = features2[:, :min_feature_dim]
            else:
                from sklearn.decomposition import PCA
                
                # Reduce both feature sets
                pca1 = PCA(n_components=target_dim)
                features1_reduced = pca1.fit_transform(features1)
                
                pca2 = PCA(n_components=target_dim)
                features2_reduced = pca2.fit_transform(features2)
                
                features1 = features1_reduced
                features2 = features2_reduced
                print(f"Reduced features to {target_dim} dimensions for correlation")
        
        # Average correlation across all feature dimensions
        correlations = []
        min_dim = min(features1.shape[1], features2.shape[1])
        
        for i in range(min_dim):
            corr = np.corrcoef(features1[:, i], features2[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        return float(np.mean(correlations)) if correlations else 0
    
    def create_tsne_visualizations(self, model_features, labels, label_names=None):
        """Create t-SNE visualizations for all models (multiclass-ready)"""
        print("Creating t-SNE visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        labels = np.asarray(labels)
        classes = np.unique(labels)
        n_classes = len(classes)
        palette = sns.color_palette("tab20", n_classes)
        class_names = [str(label_names[c]) if label_names is not None and c < len(label_names) else str(int(c)) for c in classes]
        
        for idx, (model_name, features) in enumerate(model_features.items()):
            if len(features) == 0 or idx >= 4:
                continue
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
            features_2d = tsne.fit_transform(features)
            
            # Plot
            ax = axes[idx]
            for ci, c in enumerate(classes):
                mask = labels[:len(features)] == c
                if np.any(mask):
                    ax.scatter(
                        features_2d[mask, 0], 
                        features_2d[mask, 1],
                        color=palette[ci % len(palette)], 
                        label=class_names[ci],
                        alpha=0.6,
                        s=20
                    )
            
            ax.set_title(f'{model_name.capitalize()} Model Features')
            # Limit legend size for readability
            if n_classes <= 15:
                ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'tsne_feature_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_deployment_metrics(self):
        """Analyze deployment metrics for edge devices"""
        print("\n" + "="*60)
        print("ANALYZING DEPLOYMENT METRICS")
        print("="*60)
        
        deployment_metrics = {}
        
        for model_name, model in self.models.items():
            print(f"\nAnalyzing deployment metrics for {model_name}...")
            
            # Model size
            model_size_mb = self.get_model_size(model)
            
            # FLOPs calculation
            flops, flops_formatted, params, params_formatted = self.calculate_flops(model)
            
            # Inference latency
            latency_metrics = self.measure_inference_latency(model, self.processors[model_name])
            
            # Memory usage
            memory_usage = self.measure_memory_usage(model, self.processors[model_name])
            
            deployment_metrics[model_name] = {
                "model_size_mb": round(model_size_mb, 2),
                "parameters": {
                    "total": int(params),
                    "formatted": params_formatted
                },
                "flops": {
                    "total": int(flops),
                    "formatted": flops_formatted
                },
                "latency": latency_metrics,
                "memory_usage_mb": memory_usage,
                "efficiency_score": self.calculate_efficiency_score(
                    model_size_mb, latency_metrics["mean_latency_ms"]
                )
            }
            
            print(f"✓ {model_name}: {model_size_mb:.1f}MB, {latency_metrics['mean_latency_ms']:.1f}ms")
        
        # Save deployment metrics
        self.save_results(deployment_metrics, "deployment_metrics.json")
        
        # Create comparison visualizations
        self.plot_deployment_comparison(deployment_metrics)
        
        return deployment_metrics
    
    def measure_memory_usage(self, model, processor):
        """Measure GPU memory usage with correct VideoMAE inputs"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Run a forward pass
            dummy_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
            with torch.no_grad():
                inputs = processor(dummy_frames, return_tensors="pt", do_rescale=False)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                _ = model(**inputs)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            torch.cuda.empty_cache()
            return round(peak_memory, 2)
        else:
            return 0
    
    def calculate_efficiency_score(self, size_mb, latency_ms):
        """Calculate efficiency score (higher is better)"""
        # Normalize and combine size and latency
        # This is a simple heuristic - adjust weights as needed
        size_weight = 0.4
        latency_weight = 0.6
        
        # Normalize to 0-1 scale (assuming max reasonable values)
        size_norm = min(size_mb / 1000, 1.0)  # Max 1GB
        latency_norm = min(latency_ms / 1000, 1.0)  # Max 1 second
        
        efficiency = size_weight * size_norm + latency_weight * latency_norm
        return round(1 - efficiency, 3)  # Higher score = more efficient
    
    def plot_decision_boundaries(self, boundary_analysis):
        """Plot decision boundary analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data for plotting
        models = list(boundary_analysis.keys())
        margins = [boundary_analysis[m]['decision_boundary_margin']['mean_margin'] for m in models]
        confidences = [boundary_analysis[m]['confidence_distribution']['mean_confidence'] for m in models]
        top1_accuracies = [boundary_analysis[m]['accuracy']['top1_accuracy'] * 100 for m in models]
        top5_accuracies = [boundary_analysis[m]['accuracy']['top5_accuracy'] * 100 for m in models]
        separations = [
            boundary_analysis[m]['pairwise_logit_separation']['mean_pairwise_distance']
            for m in models
        ]
        
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown'][:len(models)]
        
        # Plot 1: Top-1 and Top-5 Accuracy
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, top1_accuracies, width, label='Top-1', color=colors)
        axes[0, 0].bar(x + width/2, top5_accuracies, width, label='Top-5', color=colors, alpha=0.7)
        axes[0, 0].set_title('Classification Accuracy')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend()
        
        # Add text labels for accuracy values
        for i, (top1, top5) in enumerate(zip(top1_accuracies, top5_accuracies)):
            axes[0, 0].text(i - width/2, top1 + 1, f"{top1:.1f}%", ha='center', va='bottom', fontsize=8)
            axes[0, 0].text(i + width/2, top5 + 1, f"{top5:.1f}%", ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Decision margins
        axes[0, 1].bar(models, margins, color=colors)
        axes[0, 1].set_title('Decision Boundary Margins')
        axes[0, 1].set_ylabel('Mean Margin')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Confidence distributions
        axes[1, 0].bar(models, confidences, color=colors)
        axes[1, 0].set_title('Mean Prediction Confidence')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Class separations
        axes[1, 1].bar(models, separations, color=colors)
        axes[1, 1].set_title('Logit Space Class Separation')
        axes[1, 1].set_ylabel('Euclidean Distance')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Plot 4 is now Class separations (moved from Plot 3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'decision_boundary_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_deployment_comparison(self, deployment_metrics):
        """Plot deployment metrics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        models = list(deployment_metrics.keys())
        
        # Extract metrics
        sizes = [deployment_metrics[m]['model_size_mb'] for m in models]
        latencies = [deployment_metrics[m]['latency']['mean_latency_ms'] for m in models]
        flops = [deployment_metrics[m]['flops']['total'] / 1e9 for m in models]  # Convert to GFLOPs
        efficiency = [deployment_metrics[m]['efficiency_score'] for m in models]
        
        colors = ['blue', 'green', 'red', 'orange'][:len(models)]
        
        # Plot 1: Model sizes
        axes[0, 0].bar(models, sizes, color=colors)
        axes[0, 0].set_title('Model Size Comparison')
        axes[0, 0].set_ylabel('Size (MB)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Inference latency
        axes[0, 1].bar(models, latencies, color=colors)
        axes[0, 1].set_title('Inference Latency Comparison')
        axes[0, 1].set_ylabel('Latency (ms)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: FLOPs
        axes[1, 0].bar(models, flops, color=colors)
        axes[1, 0].set_title('Computational Complexity (GFLOPs)')
        axes[1, 0].set_ylabel('GFLOPs')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Efficiency score
        axes[1, 1].bar(models, efficiency, color=colors)
        axes[1, 1].set_title('Efficiency Score (Higher = Better)')
        axes[1, 1].set_ylabel('Efficiency Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'deployment_metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def load_sample_data(self):
        """Load sample data for analysis (multiclass-ready)"""
        print(f"Loading sample data from {TEST_CSV}...")
        
        # Read CSV without headers - format is: video_path label
        df = pd.read_csv(TEST_CSV, sep=' ', header=None, names=['video_path', 'label'])
        
        # Multiclass label mapping
        unique_labels = sorted(df['label'].unique())
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}

        # Fix video paths - ensure they point to processed_dataset/videos/
        df['video_path'] = df['video_path'].apply(lambda x: x.replace('videos\\', 'videos/').replace('videos/', 'processed_dataset/videos/'))

        # Sample videos for analysis
        sample_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
        
        video_paths = sample_df['video_path'].tolist()
        labels = [label_to_id[x] for x in sample_df['label'].tolist()]
        
        print(f"Loaded {len(sample_df)} samples across {len(unique_labels)} classes")
        
        return {
            'video_paths': video_paths,
            'labels': np.array(labels),
            'dataframe': sample_df,
            'label_to_id': label_to_id,
            'id_to_label': id_to_label,
        }
    
    def save_results(self, results, filename):
        """Save results to JSON file"""
        filepath = os.path.join(RESULTS_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {filepath}")
    
    def generate_summary_report(self, boundary_analysis, alignment_analysis, deployment_metrics):
        """Generate comprehensive summary report"""
        print("\n" + "="*60)
        print("GENERATING SUMMARY REPORT")
        print("="*60)
        
        report = {
            "analysis_summary": {
                "total_models_analyzed": len(self.models),
                "sample_size": SAMPLE_SIZE,
                "analysis_date": pd.Timestamp.now().isoformat()
            },
            "key_findings": {
                "decision_boundaries": self.summarize_boundary_findings(boundary_analysis),
                "semantic_alignment": self.summarize_alignment_findings(alignment_analysis),
                "deployment_readiness": self.summarize_deployment_findings(deployment_metrics)
            },
            "recommendations": self.generate_recommendations(boundary_analysis, alignment_analysis, deployment_metrics)
        }
        
        # Save comprehensive report
        self.save_results(report, "comprehensive_analysis_report.json")
        
        # Print summary to console
        self.print_summary_report(report)
        
        return report
    
    def summarize_boundary_findings(self, boundary_analysis):
        """Summarize decision boundary findings"""
        if not boundary_analysis:
            return "No boundary analysis available"
        
        findings = {}
        for model_name, analysis in boundary_analysis.items():
            # Include Top-1 and Top-5 accuracy in the summary
            findings[model_name] = {
                "decision_margin": analysis['decision_boundary_margin']['mean_margin'],
                "confidence": analysis['confidence_distribution']['mean_confidence'],
                "class_separation": analysis['pairwise_logit_separation']['mean_pairwise_distance'],
                "top1_accuracy": analysis['accuracy']['top1_accuracy'],
                "top5_accuracy": analysis['accuracy']['top5_accuracy']
            }
        
        return findings
    
    def summarize_alignment_findings(self, alignment_analysis):
        """Summarize semantic alignment findings"""
        if not alignment_analysis:
            return "No alignment analysis available"
        
        findings = {}
        for pair, analysis in alignment_analysis.items():
            findings[pair] = {
                "feature_similarity": analysis['feature_similarity']['mean'],
                "logit_similarity": analysis['logit_similarity']['mean'],
                "feature_correlation": analysis['feature_correlation']
            }
        
        return findings
    
    def summarize_deployment_findings(self, deployment_metrics):
        """Summarize deployment findings"""
        if not deployment_metrics:
            return "No deployment analysis available"
        
        findings = {}
        for model_name, metrics in deployment_metrics.items():
            findings[model_name] = {
                "size_mb": metrics['model_size_mb'],
                "latency_ms": metrics['latency']['mean_latency_ms'],
                "efficiency_score": metrics['efficiency_score']
            }
        
        return findings
    
    def generate_recommendations(self, boundary_analysis, alignment_analysis, deployment_metrics):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Deployment recommendations
        if deployment_metrics:
            sizes = [m['model_size_mb'] for m in deployment_metrics.values()]
            latencies = [m['latency']['mean_latency_ms'] for m in deployment_metrics.values()]
            
            min_size_model = min(deployment_metrics.keys(), key=lambda x: deployment_metrics[x]['model_size_mb'])
            min_latency_model = min(deployment_metrics.keys(), key=lambda x: deployment_metrics[x]['latency']['mean_latency_ms'])
            
            recommendations.extend([
                f"For size-constrained deployments, use {min_size_model} model ({deployment_metrics[min_size_model]['model_size_mb']:.1f}MB)",
                f"For latency-critical applications, use {min_latency_model} model ({deployment_metrics[min_latency_model]['latency']['mean_latency_ms']:.1f}ms)",
            ])
        
        # Alignment recommendations
        if alignment_analysis:
            best_alignment = max(alignment_analysis.keys(), 
                               key=lambda x: alignment_analysis[x]['feature_similarity']['mean'])
            recommendations.append(f"Best feature alignment found between {best_alignment}")
        
        # Boundary recommendations
        if boundary_analysis:
            best_margin = max(boundary_analysis.keys(),
                             key=lambda x: boundary_analysis[x]['decision_boundary_margin']['mean_margin'])
            recommendations.append(f"Most confident predictions from {best_margin} model")
            
            # Add recommendation based on Top-1 accuracy
            best_accuracy = max(boundary_analysis.keys(),
                             key=lambda x: boundary_analysis[x]['accuracy']['top1_accuracy'])
            recommendations.append(f"Best accuracy achieved by {best_accuracy} model " +
                                f"(Top-1: {boundary_analysis[best_accuracy]['accuracy']['top1_accuracy']*100:.2f}%, " +
                                f"Top-5: {boundary_analysis[best_accuracy]['accuracy']['top5_accuracy']*100:.2f}%)")
        
        return recommendations
    
    def print_summary_report(self, report):
        """Print summary report to console"""
        print("\n" + "="*80)
        print("DISTILLATION ANALYSIS SUMMARY REPORT")
        print("="*80)
        
        print(f"\nAnalysis completed on {report['analysis_summary']['analysis_date']}")
        print(f"Models analyzed: {report['analysis_summary']['total_models_analyzed']}")
        print(f"Sample size: {report['analysis_summary']['sample_size']} videos")
        
        print("\n📊 KEY FINDINGS:")
        print("-" * 40)
        
        # Deployment findings
        if 'deployment_readiness' in report['key_findings']:
            print("\n🚀 DEPLOYMENT METRICS:")
            for model, metrics in report['key_findings']['deployment_readiness'].items():
                print(f"  {model}: {metrics['size_mb']:.1f}MB, {metrics['latency_ms']:.1f}ms latency")
        
        # Accuracy findings
        if 'decision_boundaries' in report['key_findings']:
            print("\n📊 ACCURACY METRICS:")
            for model, metrics in report['key_findings']['decision_boundaries'].items():
                top1 = metrics.get('top1_accuracy', 0) * 100
                top5 = metrics.get('top5_accuracy', 0) * 100
                print(f"  {model}: Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")
        
        # Alignment findings
        if 'semantic_alignment' in report['key_findings']:
            print("\n🎯 SEMANTIC ALIGNMENT:")
            for pair, metrics in report['key_findings']['semantic_alignment'].items():
                print(f"  {pair}: {metrics['feature_similarity']:.3f} feature similarity")
        
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*80)
    
    def run_complete_analysis(self):
        """Run the complete distillation analysis"""
        print("Starting comprehensive distillation analysis...")
        print(f"Results will be saved to: {RESULTS_DIR}")
        
        # Load models
        self.load_models()
        
        if not self.models:
            print("❌ No models loaded successfully. Exiting.")
            return
        
        # Load sample data
        sample_data = self.load_sample_data()
        
        # Run analyses
        boundary_analysis = self.analyze_decision_boundaries(sample_data)
        alignment_analysis = self.analyze_semantic_alignment(sample_data)
        deployment_metrics = self.analyze_deployment_metrics()
        
        # Generate summary report
        summary_report = self.generate_summary_report(
            boundary_analysis, alignment_analysis, deployment_metrics
        )
        
        print("\n✅ Analysis complete! Check the results directory for detailed outputs.")
        return summary_report


def main():
    """Main function to run the distillation analysis"""
    analyzer = DistillationAnalyzer()
    results = analyzer.run_complete_analysis()
    return results


if __name__ == "__main__":
    main()