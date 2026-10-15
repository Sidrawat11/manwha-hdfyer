"""The model: hdfyer, a 2x SPAN S network distilled in experiments/exp16, loaded through spandrel."""

from pathlib import Path
import torch


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
