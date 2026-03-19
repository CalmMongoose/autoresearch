"""
Activation probing infrastructure for UFR structural analysis.
Captures intermediate activations and attention patterns during training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional
import json
import os
from pathlib import Path


class ActivationProbe:
    """
    Probe that captures activations from specific layers/modules.
    """
    def __init__(self, name: str, capture_every_n: int = 50):
        self.name = name
        self.capture_every_n = capture_every_n
        self.step = 0
        self.captured = []
        self.hooks = []
        
    def register_hook(self, module: nn.Module, hook_fn: Callable):
        """Register a forward hook on a module."""
        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        return handle
    
    def capture(self, tensor: torch.Tensor, metadata: Optional[Dict] = None):
        """Capture an activation tensor (stores on CPU to save VRAM)."""
        self.step += 1
        if self.step % self.capture_every_n == 0:
            # Store summary statistics rather than full tensor
            with torch.no_grad():
                stats = {
                    'mean': tensor.mean().item(),
                    'std': tensor.std().item(),
                    'min': tensor.min().item(),
                    'max': tensor.max().item(),
                    'shape': list(tensor.shape),
                    'step': self.step,
                    'metadata': metadata or {}
                }
                # If it's a reasonable size, store a sample
                if tensor.numel() <= 10000:
                    stats['sample'] = tensor.flatten()[:1000].cpu().float().numpy().tolist()
                self.captured.append(stats)
    
    def clear(self):
        """Clear captured data to free memory."""
        self.captured = []
        
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class AttentionProbe(ActivationProbe):
    """
    Specialized probe for attention patterns.
    Captures attention weights and computes statistics.
    """
    def capture_attention(self, attn_weights: torch.Tensor, layer_idx: int):
        """
        Capture attention pattern statistics.
        
        Args:
            attn_weights: [batch, heads, seq, seq] attention weights after softmax
            layer_idx: Which transformer layer this is from
        """
        with torch.no_grad():
            # Compute attention statistics
            B, H, T, _ = attn_weights.shape
            
            # Average attention per head (how much does each head attend)
            head_focus = attn_weights.sum(dim=-1).mean(dim=(0, 1))  # [seq]
            
            # Diagonal attention (self-attention strength)
            diagonal = attn_weights.diagonal(dim1=-2, dim2=-1).mean(dim=(0, 1))  # [seq]
            
            # Entropy per head
            entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1)
            mean_entropy = entropy.mean(dim=(0, 1)).mean().item()
            
            stats = {
                'layer_idx': layer_idx,
                'head_focus_mean': head_focus.mean().item(),
                'head_focus_std': head_focus.std().item(),
                'diagonal_mean': diagonal.mean().item(),
                'diagonal_std': diagonal.std().item(),
                'mean_entropy': mean_entropy,
                'max_attn': attn_weights.max().item(),
                'step': self.step,
            }
            
            self.captured.append(stats)


class ProbeManager:
    """
    Manages multiple probes across the model.
    """
    def __init__(self, output_dir: str = "probe_outputs"):
        self.probes: Dict[str, ActivationProbe] = {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.activations_cache = []
        self.attention_cache = []
        
    def register_probe(self, name: str, module: nn.Module, probe_type: str = "activation"):
        """
        Register a probe on a module.
        
        Args:
            name: Unique name for this probe
            module: The nn.Module to probe
            probe_type: "activation" or "attention"
        """
        if probe_type == "attention":
            probe = AttentionProbe(name)
            
            def hook_fn(module, input, output):
                # For attention, output is typically (attn_output, attn_weights) or just attn_output
                # This depends on the specific attention implementation
                # For nanoGPT-style attention, we need to capture from inside the module
                pass
            
            probe.register_hook(module, hook_fn)
        else:
            probe = ActivationProbe(name)
            
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    probe.capture(output, {'from_module': name})
            
            probe.register_hook(module, hook_fn)
        
        self.probes[name] = probe
        return probe
    
    def register_attention_probe(self, name: str, attn_module: nn.Module, layer_idx: int):
        """
        Register an attention probe that captures attention weights.
        Must be called with modules that have access to attention weights.
        """
        probe = AttentionProbe(name)
        self.probes[name] = probe
        return probe
    
    def capture_snapshot(self, model: nn.Module, sample_input: torch.Tensor):
        """
        Run a forward pass and capture activations from all probes.
        """
        with torch.no_grad():
            _ = model(sample_input)
        
        # Collect data from all probes
        snapshot = {}
        for name, probe in self.probes.items():
            snapshot[name] = probe.captured.copy()
        
        return snapshot
    
    def save_results(self, filename: str):
        """Save all captured probe data to disk."""
        data = {}
        for name, probe in self.probes.items():
            data[name] = probe.captured
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def clear_all(self):
        """Clear all probe data."""
        for probe in self.probes.values():
            probe.clear()
    
    def remove_all_hooks(self):
        """Remove all hooks from all probes."""
        for probe in self.probes.values():
            probe.remove_hooks()


def setup_probes_on_gpt(model: nn.Module, probe_manager: ProbeManager):
    """
    Set up probes on a GPT-style model for comprehensive structural analysis.
    
    Returns:
        Dict mapping probe names to the modules they're probing
    """
    probed_modules = {}
    
    # Probe embeddings
    if hasattr(model, 'transformer') and 'wte' in model.transformer:
        probe = probe_manager.register_probe("embeddings", model.transformer.wte)
        probed_modules['embeddings'] = model.transformer.wte
    
    # Probe each transformer block
    if hasattr(model, 'transformer') and 'h' in model.transformer:
        for i, block in enumerate(model.transformer.h):
            # Probe attention output
            if hasattr(block, 'attn'):
                probe_name = f"layer_{i}_attn"
                probe = probe_manager.register_probe(probe_name, block.attn)
                probed_modules[probe_name] = block.attn
            
            # Probe MLP output
            if hasattr(block, 'mlp'):
                probe_name = f"layer_{i}_mlp"
                probe = probe_manager.register_probe(probe_name, block.mlp)
                probed_modules[probe_name] = block.mlp
            
            # Probe full block output would need a hook on forward
    
    # Probe final output
    if hasattr(model, 'lm_head'):
        probe = probe_manager.register_probe("lm_head", model.lm_head)
        probed_modules['lm_head'] = model.lm_head
    
    return probed_modules


def compute_probe_statistics(probe_data: Dict) -> Dict:
    """
    Compute summary statistics from probe captures.
    """
    stats = {}
    
    for probe_name, captures in probe_data.items():
        if not captures:
            continue
        
        # Extract all means
        means = [c.get('mean', 0) for c in captures if 'mean' in c]
        stds = [c.get('std', 0) for c in captures if 'std' in c]
        
        if means:
            stats[f"{probe_name}_mean_avg"] = sum(means) / len(means)
            stats[f"{probe_name}_mean_std"] = (sum((m - stats[f"{probe_name}_mean_avg"]) ** 2 
                                                  for m in means) / len(means)) ** 0.5
        
        if stds:
            stats[f"{probe_name}_std_avg"] = sum(stds) / len(stds)
    
    return stats
