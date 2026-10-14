"""Shared model loading utility.

Loads RRDBNet weights from disk using position-based key mapping to handle
checkpoint naming mismatches. Called by both the profiler and the inference
engine so neither duplicates this logic.

Also loads the 2x student (SPAN S, distilled in experiments/exp16) through spandrel.
"""

from pathlib import Path
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch


def load_rrdbnet(model_path: Path) -> torch.nn.Module:
    """Builds RRDBNet, loads weights via position-based key mapping, and returns the model in eval mode on CUDA."""
    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    loadnet = torch.load(model_path, map_location=torch.device('cpu'))

    # Unwrap common checkpoint wrappers
    if isinstance(loadnet, dict) and 'params_ema' in loadnet:
        weights = loadnet['params_ema']
    elif isinstance(loadnet, dict) and 'params' in loadnet:
        weights = loadnet['params']
    else:
        weights = loadnet

    model_state = net.state_dict()
    raw_keys = list(weights.keys())
    target_keys = list(model_state.keys())

    if len(raw_keys) != len(target_keys):
        print(f"Warning: Key count mismatch! File: {len(raw_keys)}, Model: {len(target_keys)}")

    new_state_dict = {}
    for i in range(min(len(raw_keys), len(target_keys))):
        new_state_dict[target_keys[i]] = weights[raw_keys[i]]

    net.load_state_dict(new_state_dict, strict=False)
    return net.eval().to('cuda').half()


def _already_folded() -> None:
    """Stands in for Conv3XC.update_params once the convolutions have been folded."""


def fold_reparam(model: torch.nn.Module) -> int:
    """
    Folds SPAN's re-parameterised convolutions once. Returns how many were folded.
    spandrel's Conv3XC merges its three training convs into one eval conv on every forward pass;
    this does it a single time and switches the per-forward refold off. Call it on the final
    device and dtype, in eval mode: the fold is computed in the weights' own precision.
    """
    folded = 0
    for module in model.modules():
        if hasattr(module, "update_params") and hasattr(module, "eval_conv"):
            module.update_params()
            module.update_params = _already_folded
            folded += 1
    return folded


def load_student(model_path: Path) -> torch.nn.Module:
    """
    Loads a 2x student through spandrel, the way experiments/exp16/distill.py load_model does
    (to CUDA, eval, then FP16), folds its convolutions once and returns the bare network.
    """
    from spandrel import ModelLoader
    descriptor = ModelLoader().load_from_file(str(model_path))
    if descriptor.scale != 2:
        raise ValueError(f"{model_path} is a {descriptor.scale}x model, expected a 2x student")
    descriptor.to('cuda').eval()
    descriptor.half()
    fold_reparam(descriptor.model)
    return descriptor.model
