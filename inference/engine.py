"""Custom inference engine.

Owns the RRDBNet forward pass directly, replacing the Real-ESRGAN dependency.
Handles FP16 casting, tensor pre/post processing, and exposes a clean upscale
interface to the pipeline.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from core.model_loader import load_rrdbnet
from pathlib import Path
from inference.chunker import ChunkMeta

DEVICE = torch.device('cuda')
DTYPE = torch.float16


def _preprocess(image: np.ndarray) -> torch.Tensor:
    """Converts BGR uint8 HWC numpy array to RGB float16 CHW tensor on GPU."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    normalized = img_rgb.astype(np.float32) / 255.0
    transpose = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(transpose).unsqueeze(0).to(DEVICE).half()


def _pad(tensor: torch.Tensor, scale: int) -> tuple[torch.Tensor, int, int]:
    """Applies reflect padding to ensure dimensions are divisible by scale. Returns padded tensor, pad_h, pad_w."""
    mod_scale = scale
    H, W = tensor.shape[-2], tensor.shape[-1]
    pad_w = (mod_scale - (W % mod_scale)) % mod_scale
    pad_h = (mod_scale - (H % mod_scale)) % mod_scale
    output = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return (output, pad_h, pad_w)


def _unpad(tensor: torch.Tensor, pad_h: int, pad_w: int, scale: int) -> torch.Tensor:
    """Removes padding from the model output, accounting for the upscale factor."""
    output = tensor
    if pad_h == 0 and pad_w == 0:
        return tensor
    
    if pad_h > 0:
        scaled_h = pad_h * scale
        output = output[:, :, :-scaled_h,:]
    
    if pad_w > 0:
        scaled_w = pad_w * scale
        output = output[:, :, :, :-scaled_w]

    return output


def _postprocess(tensor: torch.Tensor) -> np.ndarray:
    """Clamps to [0,1], converts GPU tensor back to BGR uint8 HWC numpy array."""
    output_tensor = tensor.clamp(0, 1).to(torch.float32)
    output_image_converted = (output_tensor.squeeze(0) * 255).byte().cpu().numpy()
    output_image_transposed = np.transpose(output_image_converted, (1, 2, 0))
    img = cv2.cvtColor(output_image_transposed, cv2.COLOR_RGB2BGR)
    return img


def upscale(image: np.ndarray, model: torch.nn.Module, scale: int = 4) -> np.ndarray:
    """Upscales a single BGR uint8 image. Calls preprocess, pad, model forward, unpad, postprocess."""
    x = _preprocess(image)
    x_padded, pad_h, pad_w = _pad(x, scale)
    with torch.no_grad():
        y = model(x_padded)
    y_unpadded = _unpad(y, pad_h, pad_w, scale)
    img = _postprocess(y_unpadded)
    return img


def _is_oom(error: RuntimeError) -> bool:
    """True for PyTorch's OOM and for the driver's. With NVIDIA's "Prefer No Sysmem Fallback" it arrives as a plain CUDA RuntimeError."""
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _forward_batch(chunks: list[ChunkMeta], model: torch.nn.Module, scale: int) -> list[tuple[np.ndarray, ChunkMeta]]:
    """One forward pass over a batch of same-shape chunks."""
    preprocessed_chunks = []
    for chunk in chunks:
        tensor = _preprocess(chunk.page_slice)
        preprocessed_chunks.append(tensor)

    concated_chunks = torch.cat(preprocessed_chunks)
    chunks_padded, pad_h, pad_w = _pad(concated_chunks, scale)
    with torch.no_grad():
        upscaled = model(chunks_padded)
    upscaled_unpadded = _unpad(upscaled, pad_h, pad_w, scale)

    individual = torch.split(upscaled_unpadded, 1, dim=0)
    results = []
    for tensor, meta in zip(individual, chunks):
        results.append((_postprocess(tensor), meta))

    return results


def batch_upscale(chunks: list[ChunkMeta], model: torch.nn.Module, scale: int = 4) -> list[tuple[np.ndarray, ChunkMeta]]:
    """
    Upscales a batch of same-shape chunks in a single forward pass. 
    Preprocesses each chunk, concatenates into one tensor, runs model once, splits and postprocesses each result.
    On out-of-memory the batch is split in half and each half retried; a batch of 1 that still fails re-raises.
    Returns paired upscaled arrays and their ChunkMetas.
    """
    try:
        return _forward_batch(chunks, model, scale)
    except RuntimeError as e:
        if not _is_oom(e) or len(chunks) == 1:
            raise

    # Retry outside the except block, so the failed pass's tensors are no longer held by the traceback.
    torch.cuda.empty_cache()
    mid = len(chunks) // 2
    return batch_upscale(chunks[:mid], model, scale) + batch_upscale(chunks[mid:], model, scale)
