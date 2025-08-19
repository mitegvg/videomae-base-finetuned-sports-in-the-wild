"""
Tri-Model Asymmetric Distillation Loss Functions

This module implements the loss functions used in the tri-model distillation framework,
including feature distillation, attention distillation, and asymmetric knowledge transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class TriModelDistillationLoss(nn.Module):
    """
    Comprehensive loss function for tri-model asymmetric distillation.
    
    Combines:
    1. Classification loss (student predictions vs ground truth)
    2. Feature distillation loss (teacher -> student, assistant -> student)
    3. Attention distillation loss (teacher -> student, assistant -> student)
    4. Asymmetric knowledge transfer between teacher and assistant
    """
    
    def __init__(
        self,
        feature_distillation_weight: float = 1.0,
        attention_distillation_weight: float = 0.5,
        classification_loss_weight: float = 1.0,
        teacher_feature_weight: float = 1.0,
        assistant_feature_weight: float = 0.8,
        temperature: float = 4.0,
        hidden_layers_to_align: List[int] = [-1, -2, -3],
        classification_type: str = "multiclass"
    ):
        super().__init__()
        
        self.feature_distillation_weight = feature_distillation_weight
        self.attention_distillation_weight = attention_distillation_weight
        self.classification_loss_weight = classification_loss_weight
        self.teacher_feature_weight = teacher_feature_weight
        self.assistant_feature_weight = assistant_feature_weight
        self.temperature = temperature
        self.hidden_layers_to_align = hidden_layers_to_align
        self.classification_type = classification_type
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        
        # Multilabel loss function
        if classification_type == "multilabel":
            self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        student_outputs: Dict,
        teacher_outputs: Dict,
        assistant_outputs: Dict,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute tri-model distillation loss.
        
        Args:
            student_outputs: Dict with 'logits', 'hidden_states', 'attentions'
            teacher_outputs: Dict with 'logits', 'hidden_states', 'attentions'
            assistant_outputs: Dict with 'logits', 'hidden_states', 'attentions'
            labels: Ground truth labels
            
        Returns:
            Dict containing individual loss components and total loss
        """
        losses = {}
        
        # 1. Classification Loss
        if self.classification_type == "multilabel":
            # For multilabel, use BCE with logits loss
            classification_loss = self.bce_loss(student_outputs['logits'], labels.float())
        else:
            # For binary and multiclass, use cross entropy
            classification_loss = self.ce_loss(student_outputs['logits'], labels)
        losses['classification_loss'] = classification_loss
        
        # 2. Feature Distillation Loss
        feature_loss = self._compute_feature_distillation_loss(
            student_outputs['hidden_states'],
            teacher_outputs['hidden_states'],
            assistant_outputs['hidden_states']
        )
        losses['feature_distillation_loss'] = feature_loss
        
        # 3. Attention Distillation Loss
        if 'attentions' in student_outputs and 'attentions' in teacher_outputs:
            attention_loss = self._compute_attention_distillation_loss(
                student_outputs['attentions'],
                teacher_outputs['attentions'],
                assistant_outputs.get('attentions')
            )
            losses['attention_distillation_loss'] = attention_loss
        else:
            losses['attention_distillation_loss'] = torch.tensor(0.0, device=labels.device)
        
        # 4. Asymmetric Knowledge Transfer Loss
        asymmetric_loss = self._compute_asymmetric_knowledge_loss(
            teacher_outputs['hidden_states'],
            assistant_outputs['hidden_states']
        )
        losses['asymmetric_knowledge_loss'] = asymmetric_loss
        
        # Total Loss
        total_loss = (
            self.classification_loss_weight * classification_loss +
            self.feature_distillation_weight * feature_loss +
            self.attention_distillation_weight * losses['attention_distillation_loss'] +
            0.1 * asymmetric_loss  # Small weight for asymmetric loss
        )
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_feature_distillation_loss(
        self,
        student_hidden: List[torch.Tensor],
        teacher_hidden: List[torch.Tensor],
        assistant_hidden: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute feature distillation loss from teacher and assistant to student."""
        total_loss = 0.0
        num_layers = 0
        
        for layer_idx in self.hidden_layers_to_align:
            if abs(layer_idx) <= len(student_hidden):
                student_feat = student_hidden[layer_idx]
                teacher_feat = teacher_hidden[layer_idx]
                assistant_feat = assistant_hidden[layer_idx]
                
                # Align dimensions if necessary
                student_feat = self._align_feature_dimensions(student_feat, teacher_feat)
                assistant_feat = self._align_feature_dimensions(assistant_feat, teacher_feat)
                
                # Teacher -> Student distillation
                teacher_loss = self.mse_loss(student_feat, teacher_feat)
                
                # Assistant -> Student distillation
                assistant_loss = self.mse_loss(student_feat, assistant_feat)
                
                # Weighted combination
                layer_loss = (
                    self.teacher_feature_weight * teacher_loss +
                    self.assistant_feature_weight * assistant_loss
                )
                
                total_loss += layer_loss
                num_layers += 1
        
        return total_loss / max(num_layers, 1)
    
    def _compute_attention_distillation_loss(
        self,
        student_attentions: List[torch.Tensor],
        teacher_attentions: List[torch.Tensor],
        assistant_attentions: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute attention distillation loss with proper dimension alignment."""
        total_loss = 0.0
        num_layers = min(len(student_attentions), len(teacher_attentions))
        
        for i in range(num_layers):
            student_attn = student_attentions[i]
            teacher_attn = teacher_attentions[i]
            
            # Align attention dimensions (handle different number of heads)
            aligned_teacher_attn = self._align_attention_dimensions(student_attn, teacher_attn)
            
            # Teacher -> Student attention distillation
            teacher_loss = self.mse_loss(student_attn, aligned_teacher_attn)
            total_loss += teacher_loss
            
            # Assistant -> Student attention distillation (if available)
            if assistant_attentions and i < len(assistant_attentions):
                assistant_attn = assistant_attentions[i]
                aligned_assistant_attn = self._align_attention_dimensions(student_attn, assistant_attn)
                assistant_loss = self.mse_loss(student_attn, aligned_assistant_attn)
                total_loss += 0.5 * assistant_loss  # Lower weight for assistant attention
        
        return total_loss / max(num_layers, 1)
    
    def _compute_asymmetric_knowledge_loss(
        self,
        teacher_hidden: List[torch.Tensor],
        assistant_hidden: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute asymmetric knowledge transfer loss between teacher and assistant.
        This encourages the models to learn complementary features.
        """
        total_loss = 0.0
        num_layers = 0
        
        for layer_idx in self.hidden_layers_to_align:
            if abs(layer_idx) <= min(len(teacher_hidden), len(assistant_hidden)):
                teacher_feat = teacher_hidden[layer_idx]
                assistant_feat = assistant_hidden[layer_idx]
                
                # Align dimensions
                assistant_feat = self._align_feature_dimensions(assistant_feat, teacher_feat)
                
                # Compute cosine similarity to encourage complementary features
                teacher_norm = F.normalize(teacher_feat, p=2, dim=-1)
                assistant_norm = F.normalize(assistant_feat, p=2, dim=-1)
                
                # Cosine similarity
                cosine_sim = torch.sum(teacher_norm * assistant_norm, dim=-1)
                
                # Encourage diversity (lower similarity)
                diversity_loss = torch.mean(cosine_sim ** 2)
                total_loss += diversity_loss
                num_layers += 1
        
        return total_loss / max(num_layers, 1)
    
    def _align_feature_dimensions(
        self,
        source_feat: torch.Tensor,
        target_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Align source features to match target features dimensions.
        
        Handles:
        - Feature dimension mismatches via linear projection
        - Sequence length differences via interpolation
        - Various tensor shapes (3D, 4D)
        """
        source_shape = source_feat.shape
        target_shape = target_feat.shape
        
        # Quick return if shapes already match
        if source_shape == target_shape:
            return source_feat
        
        # Step 1: Handle feature dimension alignment (last dimension)
        if source_shape[-1] != target_shape[-1]:
            source_dim = source_feat.shape[-1]
            target_dim = target_feat.shape[-1]
            
            # Create or get linear projection layer
            if not hasattr(self, f'_proj_{source_dim}_to_{target_dim}'):
                projection = nn.Linear(source_dim, target_dim, bias=False)
                projection = projection.to(source_feat.device)
                setattr(self, f'_proj_{source_dim}_to_{target_dim}', projection)
            
            proj_layer = getattr(self, f'_proj_{source_dim}_to_{target_dim}')
            source_feat = proj_layer(source_feat)
            source_shape = source_feat.shape  # Update shape after projection
        
        # Step 2: Handle sequence length differences (only for 3D tensors)
        if len(source_shape) == 3 and len(target_shape) == 3:
            if source_shape[1] != target_shape[1]:  # Different sequence lengths
                # Transpose to [batch, hidden_dim, seq_len] for interpolation
                source_transposed = source_feat.transpose(1, 2)  # [batch, hidden_dim, seq_len]
                # Interpolate sequence length
                interpolated = F.interpolate(source_transposed, size=target_shape[1], mode='linear', align_corners=False)
                # Transpose back to [batch, seq_len, hidden_dim]
                source_feat = interpolated.transpose(1, 2)
        
        # Step 3: Handle spatial dimensions (for 4D tensors)
        elif len(source_shape) == 4 and len(target_shape) == 4:
            if source_shape[2:] != target_shape[2:]:  # Different spatial dimensions
                source_feat = F.interpolate(source_feat, size=target_shape[2:], mode='bilinear', align_corners=False)
        
        return source_feat
    
    def _align_attention_dimensions(
        self,
        student_attn: torch.Tensor,
        teacher_attn: torch.Tensor
    ) -> torch.Tensor:
        """
        Align teacher attention to match student attention dimensions.
        
        Handles:
        - Different number of attention heads (dimension 1)
        - Different sequence lengths (dimensions 2 and 3)
        
        Args:
            student_attn: Student attention tensor [batch, num_heads, seq_len, seq_len]
            teacher_attn: Teacher attention tensor [batch, num_heads, seq_len, seq_len]
            
        Returns:
            Aligned teacher attention tensor matching student dimensions
        """
        student_shape = student_attn.shape
        teacher_shape = teacher_attn.shape
        
        # Quick return if shapes already match
        if student_shape == teacher_shape:
            return teacher_attn
        
        aligned_attn = teacher_attn
        
        # Step 1: Handle different number of attention heads (dimension 1)
        if student_shape[1] != teacher_shape[1]:
            student_heads = student_shape[1]
            teacher_heads = teacher_shape[1]
            
            if teacher_heads > student_heads:
                # Average pool teacher heads to match student heads
                # Reshape to [batch, student_heads, teacher_heads//student_heads, seq_len, seq_len]
                heads_per_group = teacher_heads // student_heads
                remaining_heads = teacher_heads % student_heads
                
                if remaining_heads == 0:
                    # Perfect division - average pool
                    aligned_attn = aligned_attn.view(
                        teacher_shape[0], student_heads, heads_per_group, 
                        teacher_shape[2], teacher_shape[3]
                    )
                    aligned_attn = aligned_attn.mean(dim=2)
                else:
                    # Imperfect division - select first student_heads
                    aligned_attn = aligned_attn[:, :student_heads, :, :]
            else:
                # teacher_heads < student_heads - repeat teacher heads
                repeat_factor = student_heads // teacher_heads
                remaining = student_heads % teacher_heads
                
                if remaining == 0:
                    # Perfect division - repeat
                    aligned_attn = aligned_attn.repeat(1, repeat_factor, 1, 1)
                else:
                    # Imperfect division - repeat and concatenate partial
                    repeated = aligned_attn.repeat(1, repeat_factor, 1, 1)
                    partial = aligned_attn[:, :remaining, :, :]
                    aligned_attn = torch.cat([repeated, partial], dim=1)
        
        # Step 2: Handle different sequence lengths (dimensions 2 and 3)
        current_shape = aligned_attn.shape
        if current_shape[2] != student_shape[2] or current_shape[3] != student_shape[3]:
            # Reshape for interpolation: [batch * num_heads, seq_len, seq_len]
            batch_size, num_heads = current_shape[0], current_shape[1]
            reshaped = aligned_attn.view(batch_size * num_heads, current_shape[2], current_shape[3])
            
            # Add channel dimension for interpolation: [batch * num_heads, 1, seq_len, seq_len]
            reshaped = reshaped.unsqueeze(1)
            
            # Interpolate to target sequence length
            target_size = (student_shape[2], student_shape[3])
            interpolated = F.interpolate(reshaped, size=target_size, mode='bilinear', align_corners=False)
            
            # Remove channel dimension and reshape back: [batch, num_heads, seq_len, seq_len]
            interpolated = interpolated.squeeze(1)
            aligned_attn = interpolated.view(batch_size, num_heads, student_shape[2], student_shape[3])
        
        return aligned_attn


class ContrastiveLoss(nn.Module):
    """Contrastive loss for encouraging diverse representations between teacher and assistant."""
    
    def __init__(self, temperature: float = 0.1, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, teacher_features: torch.Tensor, assistant_features: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss to encourage diversity between teacher and assistant features.
        
        Args:
            teacher_features: Features from teacher model [batch_size, feature_dim]
            assistant_features: Features from assistant model [batch_size, feature_dim]
            
        Returns:
            Contrastive loss value
        """
        # Normalize features
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)
        assistant_norm = F.normalize(assistant_features, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.mm(teacher_norm, assistant_norm.t()) / self.temperature
        
        # Create labels (diagonal should be positive pairs, others negative)
        batch_size = teacher_features.size(0)
        labels = torch.arange(batch_size, device=teacher_features.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss
