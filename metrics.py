"""
Structural metrics for Unifying Factored Representation (UFR) research.
These metrics capture properties of model architecture that relate to
interpretability, factorization, and modularity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import numpy as np


@dataclass
class StructuralMetrics:
    """Collection of structural metrics for a model."""
    # Performance metrics
    val_bpb: float = 0.0
    
    # Redundancy / Specialization metrics
    head_consistency: float = 0.0  # Lower = heads are more distinct
    head_entropy: float = 0.0      # Entropy of attention pattern diversity
    
    # Capacity / Efficiency metrics  
    activation_entropy: float = 0.0      # Higher = more diverse activations
    effective_rank_mean: float = 0.0       # Mean effective rank of weight matrices
    max_singular_value_ratio: float = 0.0  # Ratio of max/min significant singular values
    
    # Information flow metrics
    inter_layer_mi: float = 0.0           # Mutual information between layers
    gradient_flow_variance: float = 0.0   # Variance in gradient magnitudes
    
    # Modularity metrics
    weight_sparsity: float = 0.0          # Fraction of near-zero weights
    layer_independence: float = 0.0       # How independent are layer transformations
    
    # Composite score
    composite_score: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    def __str__(self) -> str:
        lines = ["---"]
        lines.append(f"{'val_bpb':30s}: {self.val_bpb:.6f}")
        lines.append(f"{'head_consistency':30s}: {self.head_consistency:.6f}  (lower = more distinct)")
        lines.append(f"{'head_entropy':30s}: {self.head_entropy:.6f}  (higher = more focused)")
        lines.append(f"{'activation_entropy':30s}: {self.activation_entropy:.6f}  (higher = more diverse)")
        lines.append(f"{'effective_rank_mean':30s}: {self.effective_rank_mean:.6f}")
        lines.append(f"{'max_singular_value_ratio':30s}: {self.max_singular_value_ratio:.6f}")
        lines.append(f"{'inter_layer_mi':30s}: {self.inter_layer_mi:.6f}")
        lines.append(f"{'gradient_flow_variance':30s}: {self.gradient_flow_variance:.6f}")
        lines.append(f"{'weight_sparsity':30s}: {self.weight_sparsity:.6f}")
        lines.append(f"{'layer_independence':30s}: {self.layer_independence:.6f}")
        lines.append(f"{'composite_score':30s}: {self.composite_score:.6f}")
        return "\n".join(lines)


def compute_activation_entropy(activations: torch.Tensor, num_bins: int = 50) -> float:
    """
    Compute entropy of activation distribution.
    Higher entropy = more diverse activations being used.
    """
    with torch.no_grad():
        flat = activations.view(-1).float().cpu().numpy()
        
        # Compute histogram
        hist, _ = np.histogram(flat, bins=num_bins, range=(flat.min(), flat.max()))
        hist = hist + 1e-10
        hist = hist / hist.sum()
        
        # Compute entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        max_entropy = math.log2(num_bins)
        
        return entropy / max_entropy


def compute_head_consistency(attention_weights: torch.Tensor) -> float:
    """
    Measure how similar attention heads are to each other.
    Lower = heads are more distinct/specialized.
    """
    with torch.no_grad():
        if attention_weights.dim() == 4:
            # [batch, heads, seq, seq]
            B, H, T, _ = attention_weights.shape
            # Flatten per head
            heads = attention_weights.permute(1, 0, 2, 3).reshape(H, -1)
        elif attention_weights.dim() == 3:
            # [batch, seq, heads]
            B, T, H = attention_weights.shape
            heads = attention_weights.permute(2, 0, 1).reshape(H, -1)
        else:
            return 0.0
        
        # Normalize
        heads_norm = F.normalize(heads, dim=1)
        
        # Compute pairwise similarities
        similarities = heads_norm @ heads_norm.T
        
        # Average of off-diagonal
        mask = 1 - torch.eye(H, device=similarities.device)
        avg_similarity = (similarities * mask).sum() / mask.sum()
        
        return avg_similarity.item()


def compute_head_entropy(attention_weights: torch.Tensor) -> float:
    """
    Compute entropy of attention distribution per head, averaged.
    Higher = more diverse/spread attention patterns.
    """
    with torch.no_grad():
        if attention_weights.dim() != 4:
            return 0.0
        
        B, H, T, _ = attention_weights.shape
        
        # Add small epsilon for numerical stability
        weights = attention_weights + 1e-10
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Entropy per query position
        entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1)  # [B, H, T]
        
        # Max entropy is log(T)
        max_entropy = math.log(T)
        
        # Average normalized entropy
        normalized = entropy / max_entropy
        return normalized.mean().item()


def compute_effective_rank(weight_matrix: torch.Tensor, threshold: float = 0.01) -> float:
    """
    Compute effective rank by counting singular values above threshold.
    """
    with torch.no_grad():
        if weight_matrix.dim() < 2:
            return 0
        
        # Reshape to 2D
        if weight_matrix.dim() > 2:
            weight_matrix = weight_matrix.view(-1, weight_matrix.shape[-1])
        
        m, n = weight_matrix.shape
        k = min(m, n)
        
        # Use economy SVD
        try:
            U, S, V = torch.svd(weight_matrix.float())
        except:
            return 0.0
        
        # Count singular values above threshold
        max_sv = S.max()
        if max_sv == 0:
            return 0.0
        
        effective = (S / max_sv > threshold).sum().item()
        return effective


def compute_model_effective_ranks(model: nn.Module, threshold: float = 0.01) -> Dict[str, float]:
    """
    Compute effective rank statistics for all weight matrices in model.
    """
    ranks = []
    max_sv_ratios = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            rank = compute_effective_rank(module.weight, threshold)
            ranks.append(float(rank))
            
            # Compute singular value ratio
            try:
                with torch.no_grad():
                    U, S, V = torch.svd(module.weight.float())
                    if S.max() > 0:
                        top_k = max(1, len(S) // 20)
                        ratio = S[:top_k].mean() / S[-top_k:].mean()
                        max_sv_ratios.append(ratio.item())
            except:
                pass
    
    return {
        'effective_rank_mean': np.mean(ranks) if ranks else 0.0,
        'max_singular_value_ratio': np.mean(max_sv_ratios) if max_sv_ratios else 0.0,
    }


def compute_weight_sparsity(model: nn.Module, threshold: float = 1e-4) -> float:
    """
    Compute fraction of weights below threshold (effective sparsity).
    """
    total = 0
    near_zero = 0
    
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:  # Only matrix parameters
                total += p.numel()
                near_zero += (p.abs() < threshold).sum().item()
    
    return near_zero / total if total > 0 else 0.0


def compute_gradient_flow_variance(model: nn.Module) -> float:
    """
    Compute variance in gradient magnitudes across layers.
    """
    grad_norms = []
    
    for name, p in model.named_parameters():
        if p.grad is not None and p.dim() >= 2:
            grad_norms.append(p.grad.norm().item())
    
    if len(grad_norms) < 2:
        return 0.0
    
    return np.var(grad_norms)


def compute_composite_score(
    val_bpb: float,
    head_consistency: float,
    head_entropy: float,
    activation_entropy: float,
    effective_rank_mean: float,
    weight_sparsity: float,
    baseline_bpb: float = 1.0,
    weights: Dict[str, float] = None
) -> float:
    """
    Compute composite score balancing performance and structural properties.
    
    Args:
        val_bpb: Validation bits per byte
        head_consistency: Lower is better (heads more distinct)
        head_entropy: Higher is better (more focused attention)
        activation_entropy: Higher is better (more diverse activations)
        effective_rank_mean: Higher is better (more capacity used)
        weight_sparsity: Lower is better (less sparsity = more used)
        baseline_bpb: Baseline for normalization
        weights: Custom weights for each component
    
    Returns:
        Composite score (higher is better)
    """
    if weights is None:
        weights = {
            'performance': 0.50,
            'head_consistency': 0.15,
            'head_entropy': 0.05,
            'activation_entropy': 0.10,
            'effective_rank': 0.10,
            'weight_sparsity': 0.10,
        }
    
    # Normalize performance (lower bpb = better)
    performance = baseline_bpb / max(val_bpb, 0.1)
    
    # Head consistency: lower is better, so invert
    head_cons_score = 1.0 - head_consistency
    
    # Weight sparsity: lower is better, so invert
    sparsity_score = 1.0 - weight_sparsity
    
    # Normalize effective rank (assume max ~768 for reference)
    rank_score = min(effective_rank_mean / 500, 1.0)
    
    score = (
        weights['performance'] * performance +
        weights['head_consistency'] * head_cons_score +
        weights['head_entropy'] * head_entropy +
        weights['activation_entropy'] * activation_entropy +
        weights['effective_rank'] * rank_score +
        weights['weight_sparsity'] * sparsity_score
    )
    
    return score


def analyze_model_structure(
    model: nn.Module,
    val_bpb: float,
    sample_input: Optional[torch.Tensor] = None,
    baseline_bpb: float = 1.0
) -> StructuralMetrics:
    """
    Complete structural analysis of a model.
    
    Args:
        model: The model to analyze
        val_bpb: Validation BPB score
        sample_input: Optional sample input for activation analysis
        baseline_bpb: Baseline val_bpb for score normalization
    
    Returns:
        StructuralMetrics object with all metrics
    """
    metrics = StructuralMetrics()
    metrics.val_bpb = val_bpb
    
    # Compute effective ranks
    rank_stats = compute_model_effective_ranks(model)
    metrics.effective_rank_mean = rank_stats['effective_rank_mean']
    metrics.max_singular_value_ratio = rank_stats['max_singular_value_ratio']
    
    # Compute weight sparsity
    metrics.weight_sparsity = compute_weight_sparsity(model)
    
    # Compute gradient flow variance
    metrics.gradient_flow_variance = compute_gradient_flow_variance(model)
    
    # Compute composite score (without activations for now)
    metrics.composite_score = compute_composite_score(
        val_bpb=val_bpb,
        head_consistency=0.0,  # Will be filled if hooks are available
        head_entropy=0.0,
        activation_entropy=0.0,
        effective_rank_mean=metrics.effective_rank_mean,
        weight_sparsity=metrics.weight_sparsity,
        baseline_bpb=baseline_bpb
    )
    
    return metrics
