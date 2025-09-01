"""
Tri-Model Asymmetric Distillation Loss Functions

This module implements the loss functions used in the tri-model distillation framework,
including feature distillation, attention distillation, and asymmetric knowledge transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

# Helper: safe softmax over head dimension
def _head_softmax(x: torch.Tensor, dim: int = 1):
    return torch.softmax(x, dim=dim)


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
        # Set to None to force explicit passing from config / trainer; prevents accidental hardcoded defaults.
        feature_distillation_weight: Optional[float] = None,
        attention_distillation_weight: Optional[float] = None,
        logits_distillation_weight: Optional[float] = None,
        classification_loss_weight: Optional[float] = None,
        teacher_feature_weight: Optional[float] = None,
        assistant_feature_weight: Optional[float] = None,
        teacher_logits_weight: Optional[float] = None,
        assistant_logits_weight: Optional[float] = None,
        teacher_attention_weight: Optional[float] = None,
        assistant_attention_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        logits_temperature: Optional[float] = None,
        hidden_layers_to_align: Optional[List[int]] = None,
        classification_type: str = "multiclass"
    ):
        super().__init__()
        # Validate required args provided
        required_pairs = [
            ("feature_distillation_weight", feature_distillation_weight),
            ("attention_distillation_weight", attention_distillation_weight),
            ("logits_distillation_weight", logits_distillation_weight),
            ("classification_loss_weight", classification_loss_weight),
            ("teacher_feature_weight", teacher_feature_weight),
            ("assistant_feature_weight", assistant_feature_weight),
            ("teacher_logits_weight", teacher_logits_weight),
            ("assistant_logits_weight", assistant_logits_weight),
            ("temperature", temperature),
            ("logits_temperature", logits_temperature),
            ("teacher_attention_weight", teacher_attention_weight),
            ("assistant_attention_weight", assistant_attention_weight),
        ]
        missing = [n for n, v in required_pairs if v is None]
        if missing:
            raise ValueError(f"TriModelDistillationLoss missing required explicit args: {missing}")

        # Core weights
        self.feature_distillation_weight = float(feature_distillation_weight)
        self.attention_distillation_weight = float(attention_distillation_weight)
        self.logits_distillation_weight = float(logits_distillation_weight)
        self.classification_loss_weight = float(classification_loss_weight)
        self.teacher_feature_weight = float(teacher_feature_weight)
        self.assistant_feature_weight = float(assistant_feature_weight)
        self.teacher_logits_weight = float(teacher_logits_weight)
        self.assistant_logits_weight = float(assistant_logits_weight)
        self.teacher_attention_weight = float(teacher_attention_weight)
        self.assistant_attention_weight = float(assistant_attention_weight)
        self.temperature = float(temperature)
        self.logits_temperature = float(logits_temperature)
        self.hidden_layers_to_align = hidden_layers_to_align if hidden_layers_to_align is not None else [-1]
        self.classification_type = classification_type

        # Advanced flags (populated externally by trainer)
        self.enable_layer_fusion = False
        self.layer_fusion_mode = "attention"
        self.layer_fusion_teacher_layers = []
        self.layer_fusion_assistant_layers = []
        self.fusion_source = "teacher"
        self.fusion_source_weighting = "learned"
        self.fusion_teacher_weight = 0.5
        self.fusion_assistant_weight = 0.5
        self.fusion_projection_dim = None
        self.temporal_delta_distillation_weight = 0.0
        self.temporal_delta_layers = [-1]

        # Attention head squeeze controls
        self.attention_head_squeeze = False
        self.attention_head_squeeze_mode = "learned"  # 'learned' | 'avg'
        self.head_squeeze_ortho_weight = 1e-3

        # Caches / lazy params
        self._fusion_attention = None
        self._head_squeeze_layer = None  # type: Optional[nn.Module]
        self._head_mix_proj = None  # P in R^{H_a x H_s}

        # Loss primitives
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        if self.classification_type == "multilabel":
            self.bce_loss = nn.BCEWithLogitsLoss()
        # Logits KD gating defaults (can be overridden via config/trainer)
        self.kd_conf_threshold = 0.5  # τ
        self.kd_min_keep_ratio = 0.25
        self.kd_dynamic_lower = True
        self._last_kd_keep_ratio = torch.tensor(1.0)
        self._last_kd_conf_threshold = torch.tensor(self.kd_conf_threshold)
        self._last_kd_effective = torch.tensor(0.0)
    
    def forward(
        self,
        student_outputs,
        teacher_outputs=None,
        assistant_outputs=None,
        labels=None,
        hidden_layers_to_align=None
    ):
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
        if hidden_layers_to_align is None:
            hidden_layers_to_align = self.hidden_layers_to_align  # attribute set in __init__
        
        losses = {}
        
        # 1. Classification Loss
        if self.classification_type == "multilabel":
            # For multilabel, use BCE with logits loss
            classification_loss = self.bce_loss(student_outputs['logits'], labels.float())
        else:
            # For binary and multiclass, use cross entropy
            classification_loss = self.ce_loss(student_outputs['logits'], labels)
        losses['classification_loss'] = classification_loss
        
        # 2. Feature Distillation Loss (only if active and tensors exist)
        if self.feature_distillation_weight > 0 and student_outputs.get('hidden_states') is not None:
            feature_loss = self._compute_feature_distillation_loss(
                student_outputs['hidden_states'],
                teacher_outputs['hidden_states'],
                assistant_outputs['hidden_states']
            )
        else:
            feature_loss = torch.zeros((), device=student_outputs['logits'].device)
        losses['feature_distillation_loss'] = feature_loss

        # 2b. Temporal Delta (motion-aware) Distillation
        if self.temporal_delta_distillation_weight > 0 and student_outputs.get('hidden_states') is not None:
            delta_loss = self._compute_temporal_delta_loss(
                student_outputs['hidden_states'],
                teacher_outputs['hidden_states']
            )
        else:
            delta_loss = torch.zeros((), device=student_outputs['logits'].device)
        losses['temporal_delta_distillation_loss'] = delta_loss
        
        # 3. Attention Distillation Loss (only if active)
        if self.attention_distillation_weight > 0 and \
           student_outputs.get('attentions') is not None and \
           teacher_outputs.get('attentions') is not None:
            attention_loss = self._compute_attention_distillation_loss(
                student_outputs['attentions'],
                teacher_outputs['attentions'],
                assistant_outputs.get('attentions')
            )
        else:
            attention_loss = torch.zeros((), device=student_outputs['logits'].device)
        losses['attention_distillation_loss'] = attention_loss
        
        # 4. Logits Distillation Loss (only if active)
        if self.logits_distillation_weight > 0:
            logits_raw, kd_logs = self._compute_logits_distillation_loss(
                student_outputs['logits'],
                teacher_outputs['logits'],
                assistant_outputs['logits']
            )
            logits_loss = self.logits_distillation_weight * logits_raw
            losses.update({k: v for k, v in kd_logs.items() if isinstance(v, torch.Tensor)})
        else:
            logits_loss = torch.zeros((), device=student_outputs['logits'].device)
        losses['logits_distillation_loss'] = logits_loss
        
        # 5. Asymmetric Knowledge Transfer Loss (only if hidden states present)
        if self.feature_distillation_weight > 0 and student_outputs.get('hidden_states') is not None:
            asymmetric_loss = self._compute_asymmetric_knowledge_loss(
                teacher_outputs['hidden_states'],
                assistant_outputs['hidden_states']
            )
        else:
            asymmetric_loss = torch.zeros((), device=student_outputs['logits'].device)
        losses['asymmetric_knowledge_loss'] = asymmetric_loss
        
        # Total Loss
        total_loss = (
            self.classification_loss_weight * classification_loss +
            self.feature_distillation_weight * feature_loss +
            self.attention_distillation_weight * attention_loss +
            self.logits_distillation_weight * logits_loss +
            0.1 * asymmetric_loss +
            self.temporal_delta_distillation_weight * delta_loss
        )
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_feature_distillation_loss(
        self,
        student_hidden: List[torch.Tensor],
        teacher_hidden: List[torch.Tensor],
        assistant_hidden: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute feature distillation loss (cosine distance) from teacher & assistant to student.

        Replaces raw MSE (scale O(10-100)) with cosine distance (1 - cos) averaged over tokens, yielding
        raw values ~0.05-0.25 early for stable weighting. No config changes required.
        """
        device = student_hidden[0].device if student_hidden else (
            teacher_hidden[0].device if teacher_hidden else assistant_hidden[0].device
        )
        teacher_w = float(self.teacher_feature_weight)
        assistant_w = float(self.assistant_feature_weight)
        denom_w = max(teacher_w + assistant_w, 1e-8)

        def _prep(feat: torch.Tensor) -> torch.Tensor:
            # Collapse all non-batch, non-hidden dims into a token dimension
            if feat.dim() == 3:  # [B, S, C]
                return feat
            B = feat.shape[0]; C = feat.shape[-1]
            return feat.view(B, -1, C)

        def _cosine_loss(student_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
            if student_feat.shape != target_feat.shape:
                target_feat_aligned = self._align_feature_dimensions(target_feat, student_feat)
            else:
                target_feat_aligned = target_feat
            s = _prep(student_feat)
            t = _prep(target_feat_aligned)
            s = F.normalize(s, p=2, dim=-1)
            t = F.normalize(t, p=2, dim=-1)
            cos = (s * t).sum(-1)  # [B, N]
            dist = 1.0 - cos       # [B, N]
            return dist.mean()

        total_loss = torch.tensor(0.0, device=device)
        num_layers = 0

        if self.enable_layer_fusion:
            # Build fused target representation(s) similar to previous logic, then cosine distance per selected layer.
            fused_targets = []
            def fuse_layers(hidden_list: List[torch.Tensor], indices: List[int], cache_attr: str):
                h_len = len(hidden_list)
                if not indices:
                    indices_use = list(range(h_len))
                else:
                    indices_use = [i if i >= 0 else h_len + i for i in indices if abs(i) < h_len]
                tensor_stack = torch.stack([hidden_list[i] for i in indices_use], dim=0)  # [L,B,S,H]
                if self.layer_fusion_mode == 'attention':
                    attn_param_name = f"_fusion_attn_{cache_attr}"
                    attn_param = getattr(self, attn_param_name, None)
                    if attn_param is None or attn_param.numel() != len(indices_use):
                        attn_param = nn.Parameter(torch.zeros(len(indices_use), device=tensor_stack.device))
                        nn.init.normal_(attn_param, std=0.02)
                        setattr(self, attn_param_name, attn_param)
                    elif attn_param.device != tensor_stack.device:
                        new_param = nn.Parameter(attn_param.to(tensor_stack.device))
                        setattr(self, attn_param_name, new_param)
                        attn_param = new_param
                    attn_weights = torch.softmax(attn_param, dim=0)
                    fused = torch.sum(attn_weights.view(-1,1,1,1)*tensor_stack, dim=0)
                else:
                    fused = tensor_stack.mean(dim=0)
                return fused  # [B,S,H]
            if self.fusion_source in ("teacher", "both") and teacher_hidden is not None:
                fused_teacher = fuse_layers(teacher_hidden, self.layer_fusion_teacher_layers, 'teacher')
                fused_targets.append(('teacher', fused_teacher))
            if self.fusion_source in ("assistant", "both") and assistant_hidden is not None:
                fused_assistant = fuse_layers(assistant_hidden, self.layer_fusion_assistant_layers, 'assistant')
                fused_targets.append(('assistant', fused_assistant))
            if len(fused_targets) == 2:
                if self.fusion_source_weighting == 'learned':
                    if not hasattr(self, '_fusion_source_logits'):
                        self._fusion_source_logits = nn.Parameter(torch.zeros(2, device=fused_targets[0][1].device))
                        nn.init.normal_(self._fusion_source_logits, std=0.02)
                    elif self._fusion_source_logits.device != fused_targets[0][1].device:
                        self._fusion_source_logits = nn.Parameter(self._fusion_source_logits.to(fused_targets[0][1].device))
                    src_weights = torch.softmax(self._fusion_source_logits, dim=0)
                else:
                    src_weights = torch.tensor([self.fusion_teacher_weight, self.fusion_assistant_weight], device=fused_targets[0][1].device)
                fused_final = src_weights[0]*fused_targets[0][1] + src_weights[1]*fused_targets[1][1]
            else:
                fused_final = fused_targets[0][1]
            if self.fusion_projection_dim is not None and fused_final.shape[-1] != self.fusion_projection_dim:
                key = f"_fusion_proj_{fused_final.shape[-1]}_{self.fusion_projection_dim}"
                if not hasattr(self, key):
                    setattr(self, key, nn.Linear(fused_final.shape[-1], self.fusion_projection_dim, bias=False))
                proj = getattr(self, key).to(fused_final.device)
                fused_final = proj(fused_final)
            for layer_idx in self.hidden_layers_to_align:
                if abs(layer_idx) <= len(student_hidden):
                    s_feat = student_hidden[layer_idx]
                    aligned_target = self._align_feature_dimensions(fused_final, s_feat)
                    layer_loss = _cosine_loss(s_feat, aligned_target)
                    total_loss += layer_loss
                    num_layers += 1
        else:
            for layer_idx in self.hidden_layers_to_align:
                if abs(layer_idx) <= len(student_hidden):
                    s_feat = student_hidden[layer_idx]
                    t_feat = teacher_hidden[layer_idx]
                    a_feat = assistant_hidden[layer_idx]
                    t_feat = self._align_feature_dimensions(t_feat, s_feat)
                    a_feat = self._align_feature_dimensions(a_feat, s_feat)
                    teacher_loss = _cosine_loss(s_feat, t_feat)
                    assistant_loss = _cosine_loss(s_feat, a_feat)
                    blended = (teacher_w * teacher_loss + assistant_w * assistant_loss) / denom_w
                    total_loss += blended
                    num_layers += 1
        return total_loss / max(num_layers, 1)

    def _compute_temporal_delta_loss(
        self,
        student_hidden: List[torch.Tensor],
        teacher_hidden: List[torch.Tensor]
    ) -> torch.Tensor:
        """Delta (motion) distillation: match temporal differences Δh_t = h_t - h_{t-1}."""
        device = student_hidden[0].device
        total = torch.tensor(0.0, device=device)
        count = 0
        for layer_idx in self.temporal_delta_layers:
            if abs(layer_idx) <= min(len(student_hidden), len(teacher_hidden)):
                s_feat = student_hidden[layer_idx]  # [B, T, H] typically or [B, S, H]
                t_feat = teacher_hidden[layer_idx]
                if s_feat.dim() < 3 or t_feat.dim() < 3:
                    continue  # Skip if no temporal dimension
                # Ensure dims align
                t_feat = self._align_feature_dimensions(t_feat, s_feat)
                # Compute deltas along sequence dimension (assume dim=1)
                s_delta = s_feat[:, 1:] - s_feat[:, :-1]
                t_delta = t_feat[:, 1:] - t_feat[:, :-1]
                # Align lengths if mismatch
                if s_delta.shape != t_delta.shape:
                    t_delta = self._align_feature_dimensions(t_delta, s_delta)
                delta_loss = self.mse_loss(s_delta, t_delta)
                total += delta_loss
                count += 1
        return total / max(count, 1)
    
    def _compute_logits_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        assistant_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Logits distillation with low-confidence masking & adaptive threshold.

        Mask rule (multiclass/binary): keep sample if
          max(assistant_probs) >= kd_conf_threshold OR (teacher top-1 == assistant top-1)
        If keep ratio < kd_min_keep_ratio and kd_dynamic_lower: reduce threshold by 10% (in-place) and recompute mask.
        Returns raw blended KD loss (before global logits_distillation_weight) and log stats.
        """
        logs: Dict[str, torch.Tensor] = {}
        device = student_logits.device
        # Multilabel path: keep simple (no masking for now)
        if self.classification_type == "multilabel":
            teacher_loss = self.mse_loss(student_logits, teacher_logits)
            assistant_loss = self.mse_loss(student_logits, assistant_logits)
            total_logits_loss = (
                self.teacher_logits_weight * teacher_loss +
                self.assistant_logits_weight * assistant_loss
            )
            logs["kd_mask_keep_ratio"] = torch.tensor(1.0, device=device)
            logs["kd_conf_threshold"] = torch.tensor(self.kd_conf_threshold, device=device)
            return total_logits_loss, logs

        T = self.logits_temperature if self.logits_temperature > 0 else 1.0
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits / T, dim=-1) if teacher_logits is not None else None
            assistant_probs = F.softmax(assistant_logits / T, dim=-1) if assistant_logits is not None else None
            mask = None
            keep_ratio = torch.tensor(1.0, device=device)
            if teacher_probs is not None and assistant_probs is not None:
                a_conf, a_top = assistant_probs.max(dim=-1)
                t_top = teacher_probs.argmax(dim=-1)
                agree = (a_top == t_top)
                mask = (a_conf >= self.kd_conf_threshold) | agree
                keep_ratio = mask.float().mean()
                if self.kd_dynamic_lower and keep_ratio < self.kd_min_keep_ratio and self.kd_conf_threshold > 0.05:
                    self.kd_conf_threshold *= 0.9  # relax by 10%
                    # recompute with relaxed threshold
                    mask = (a_conf >= self.kd_conf_threshold) | agree
                    keep_ratio = mask.float().mean()
                self._last_kd_keep_ratio = keep_ratio.detach()
                self._last_kd_conf_threshold = torch.tensor(self.kd_conf_threshold, device=device)
            else:
                self._last_kd_keep_ratio = torch.tensor(1.0, device=device)
                self._last_kd_conf_threshold = torch.tensor(self.kd_conf_threshold, device=device)

        student_log_probs = F.log_softmax(student_logits / T, dim=-1)

        def _kl(p_target, log_q):
            return (p_target * (p_target.clamp_min(1e-12).log() - log_q)).sum(dim=-1)

        kd_terms = []
        weights = []

        if teacher_logits is not None and self.teacher_logits_weight > 0:
            kt = _kl(teacher_probs, student_log_probs)  # [B]
            if mask is not None:
                kt = kt[mask]
            if kt.numel() > 0:
                kd_terms.append(kt.mean())
                weights.append(self.teacher_logits_weight)

        if assistant_logits is not None and self.assistant_logits_weight > 0:
            ka = _kl(assistant_probs, student_log_probs)  # [B]
            if mask is not None:
                ka = ka[mask]
            if ka.numel() > 0:
                kd_terms.append(ka.mean())
                weights.append(self.assistant_logits_weight)

        if not kd_terms:
            raw = torch.tensor(0.0, device=device)
        else:
            w = torch.tensor(weights, device=device, dtype=kd_terms[0].dtype)
            stacked = torch.stack(kd_terms)
            raw = (stacked * w).sum() / w.sum()
            raw = raw * (T * T)

        self._last_kd_effective = raw.detach()
        logs.update({
            "kd_mask_keep_ratio": self._last_kd_keep_ratio,
            "kd_conf_threshold": torch.tensor(self.kd_conf_threshold, device=device),
            "kd_logits_raw": self._last_kd_effective,
        })
        return raw, logs
    
    def _compute_attention_distillation_loss(
        self,
        student_attentions: List[torch.Tensor],
        teacher_attentions: List[torch.Tensor],
        assistant_attentions: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute attention distillation loss with moderate scaling.

        Previous version multiplied per-layer MSE by (L*L), making losses huge (O(10–100)).
        Now we scale only by L (sequence length) to lift the tiny raw MSE (~1e-4) into a usable range (~1e-2).
        """
        if not student_attentions or not teacher_attentions:
            device = student_attentions[0].device if student_attentions else (
                teacher_attentions[0].device if teacher_attentions else torch.device('cpu')
            )
            return torch.zeros((), device=device)

        device = student_attentions[0].device
        num_layers = min(len(student_attentions), len(teacher_attentions))
        total_loss = torch.tensor(0.0, device=device)

        teacher_w = float(getattr(self, 'teacher_attention_weight', 1.0))
        assistant_w = float(getattr(self, 'assistant_attention_weight', 0.0))
        denom = max(teacher_w + assistant_w, 1e-8)

        for i in range(num_layers):
            student_attn = student_attentions[i]          # [B, Hs, L, L]
            teacher_attn = teacher_attentions[i]

            aligned_teacher_attn = self._align_attention_dimensions(student_attn, teacher_attn)
            L = student_attn.shape[-1]

            # Base MSE (already mean over all dims) then mild scaling by L (NOT L*L)
            teacher_loss = self.mse_loss(student_attn, aligned_teacher_attn) * L

            assistant_loss = None
            if assistant_attentions and i < len(assistant_attentions):
                assistant_attn = assistant_attentions[i]
                if (self.attention_head_squeeze and assistant_attn.shape[1] > student_attn.shape[1]
                        and self.attention_head_squeeze_mode in {"learned", "avg"}):
                    assistant_attn = self._apply_head_squeeze_projection(assistant_attn, target_heads=student_attn.shape[1])
                aligned_assistant_attn = self._align_attention_dimensions(student_attn, assistant_attn)
                assistant_loss = self.mse_loss(student_attn, aligned_assistant_attn) * L

            if assistant_loss is not None:
                blended = (teacher_w * teacher_loss + assistant_w * assistant_loss) / denom
            else:
                blended = teacher_loss  # only teacher term available

            total_loss += blended

        # Orthogonality penalty unchanged
        if self._head_mix_proj is not None and self.head_squeeze_ortho_weight > 0:
            total_loss += self._orthogonality_penalty(self._head_mix_proj) * self.head_squeeze_ortho_weight

        return total_loss / max(num_layers, 1)

    # --------------------- Head Squeeze Helpers ---------------------
    def _maybe_init_head_mix(self, assistant_heads: int, student_heads: int, device: torch.device):
        """Lazy-init linear mixing projection P in R^{H_a x H_s}."""
        if self._head_mix_proj is None or self._head_mix_proj.shape != (assistant_heads, student_heads):
            init = torch.randn(assistant_heads, student_heads, device=device) * 0.02
            self._head_mix_proj = nn.Parameter(init)

    def _apply_head_squeeze_projection(self, attn: torch.Tensor, target_heads: int) -> torch.Tensor:
        """Apply head squeeze (avg or learned) to reduce attention heads.

        attn: [B, H_a, L, L]
        Returns: [B, target_heads, L, L]
        """
        B, H_a, L, _ = attn.shape
        if H_a == target_heads:
            return attn
        if self.attention_head_squeeze_mode == 'avg':
            group = H_a // target_heads
            trimmed = attn[:, :group * target_heads]
            return trimmed.view(B, target_heads, group, L, L).mean(2)
        # learned projection
        self._maybe_init_head_mix(H_a, target_heads, attn.device)
        # reshape to apply linear mix over head dimension via matmul
        # attn: [B,H_a,L,L] -> [B,L,L,H_a]
        attn_reordered = attn.permute(0, 2, 3, 1)
        # matmul: (B,L,L,H_a) x (H_a,H_s) -> (B,L,L,H_s)
        mixed_llhs = torch.matmul(attn_reordered, self._head_mix_proj)
        # back to [B,H_s,L,L]
        mixed = mixed_llhs.permute(0, 3, 1, 2).contiguous()
        return mixed

    def _orthogonality_penalty(self, P: torch.Tensor) -> torch.Tensor:
        """Orthogonality penalty || P^T P - I ||_F^2 encouraging diverse head combinations."""
        gram = P.t() @ P  # [H_s, H_s]
        I = torch.eye(gram.shape[0], device=gram.device)
        return ((gram - I) ** 2).mean()
    
    def _compute_asymmetric_knowledge_loss(
        self,
        teacher_hidden: List[torch.Tensor],
        assistant_hidden: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute asymmetric knowledge transfer loss between teacher and assistant.
        This encourages the models to learn complementary features.
        """
        # Get device from available tensors
        device = teacher_hidden[0].device if len(teacher_hidden) > 0 else assistant_hidden[0].device
        total_loss = torch.tensor(0.0, device=device)  # Fix: tensor instead of float
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
        
        # Step 1: Handle different number of attention heads (dimension 1) with optional squeezing
        if student_shape[1] != teacher_shape[1]:
            student_heads = student_shape[1]
            teacher_heads = teacher_shape[1]
            if self.attention_head_squeeze and teacher_heads > student_heads:
                if self.attention_head_squeeze_mode == 'avg':
                    # Average group heads
                    group = teacher_heads // student_heads
                    aligned_attn = aligned_attn[:, :group*student_heads]
                    aligned_attn = aligned_attn.view(teacher_shape[0], student_heads, group, teacher_shape[2], teacher_shape[3]).mean(2)
                else:  # learned projection across heads
                    # Flatten heads as channels and apply 1x1 conv
                    B, H, S1, S2 = aligned_attn.shape
                    # Create projection layer lazily
                    if self._head_squeeze_layer is None or getattr(self._head_squeeze_layer, 'in_heads', None) != H or getattr(self._head_squeeze_layer, 'out_heads', None) != student_heads:
                        layer = nn.Conv2d(H, student_heads, kernel_size=1, bias=False)
                        layer.in_heads = H; layer.out_heads = student_heads
                        self._head_squeeze_layer = layer.to(aligned_attn.device)
                    aligned_attn = self._head_squeeze_layer(aligned_attn)  # [B, student_heads, S1, S2]
            else:
                # Fallback pooling / repeating logic
                if teacher_heads > student_heads:
                    heads_per_group = teacher_heads // student_heads
                    remaining_heads = teacher_heads % student_heads
                    if remaining_heads == 0:
                        aligned_attn = aligned_attn.view(
                            teacher_shape[0], student_heads, heads_per_group,
                            teacher_shape[2], teacher_shape[3]
                        ).mean(dim=2)
                    else:
                        aligned_attn = aligned_attn[:, :student_heads]
                else:
                    repeat_factor = student_heads // teacher_heads
                    remaining = student_heads % teacher_heads
                    if remaining == 0:
                        aligned_attn = aligned_attn.repeat(1, repeat_factor, 1, 1)
                    else:
                        repeated = aligned_attn.repeat(1, repeat_factor, 1, 1)
                        partial = aligned_attn[:, :remaining]
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

def compute_attention_distillation_loss(
    student_attns_assistant, assistant_attns,
    student_attns_teacher=None, teacher_attns=None,
    assistant_weight=1.0, teacher_weight=0.0,
    multiply_by_heads=False, distance="mse"
):
    """
    student_attns_* / *_attns: lists of attention tensors [B, H, L, L] AFTER any head squeeze/alignment.
    Returns blended scalar loss and component dict.
    """
    import torch
    import torch.nn.functional as F
    eps = 1e-12

    def _mse(a, b):
        per = []
        for sa, sb in zip(a, b):
            if sa.shape != sb.shape:
                raise ValueError(f"Attention shape mismatch {sa.shape} vs {sb.shape}")
            # sa, sb assumed probabilities (softmaxed). MSE mean reduces over all dims.
            base = F.mse_loss(sa, sb, reduction='mean')
            L = sa.shape[-1]
            H = sa.shape[1]
            scale = (L * L)
            if multiply_by_heads:
                scale *= H
            per.append(base * scale)
        return torch.stack(per).mean() if per else torch.tensor(0.0, device=a[0].device if a else 'cpu')

    def _kl(a, b):
        per = []
        for sa, sb in zip(a, b):
            # sa, sb: probs. KL(teacher||student) = sum p * log(p / q)
            # We compute mean over batch, heads, tokens then rescale like MSE.
            L = sa.shape[-1]
            H = sa.shape[1]
            kl = (sb * (sb.add(eps).log() - sa.add(eps).log())).mean()
            scale = (L * L)
            if multiply_by_heads:
                scale *= H
            per.append(kl * scale)
        return torch.stack(per).mean() if per else torch.tensor(0.0, device=a[0].device if a else 'cpu')

    reducer = _mse if distance == "mse" else _kl

    L_assist = reducer(student_attns_assistant, assistant_attns) if assistant_attns else torch.tensor(0.0)
    L_teacher = reducer(student_attns_teacher, teacher_attns) if (teacher_attns and student_attns_teacher) else torch.tensor(0.0)

    denom = assistant_weight + teacher_weight + eps
    blended = (assistant_weight * L_assist + teacher_weight * L_teacher) / denom

    return blended, {
        "attn_loss_assistant": L_assist.detach(),
        "attn_loss_teacher": L_teacher.detach(),
        "attn_loss_blended": blended.detach(),
    }
