"""Custom inference engine.

Owns the RRDBNet forward pass directly, replacing the Real-ESRGAN dependency.
Handles FP16 casting, tensor pre/post processing, and exposes a clean upscale
interface to the pipeline.

The forward pass has three runtimes, chosen once with select_backend():
  trt       TensorRT FP16 engines (inference/trt_backend.py), fixed batch size per engine
  compiled  torch.compile on a channels_last copy of the model
  eager     plain PyTorch FP16
trt and compiled only handle each width's standard strip shape; every other shape runs
eager, and so does any shape whose engine fails to build or run. Pre/post-processing is
shared, so the output path is identical across backends.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from core.model_loader import load_rrdbnet
from pathlib import Path
from inference import trt_backend
from inference.chunker import ChunkMeta, get_strip_height

DEVICE = torch.device('cuda')
DTYPE = torch.float16
SCALE = 4

BACKENDS = ("auto", "trt", "compiled", "eager")
BACKEND = "auto"  # default for select_backend()


class _Runtime:
    """What select_backend() chose. Until it is called everything runs eager."""
    backend = "eager"
    benchmark_map = None
    trt = None           # trt_backend.TrtBackend
    compiled = None      # torch.compile'd channels_last copy of the model
    failed_shapes = set()   # (h, w) chunk shapes that fell back to eager for the rest of the run
    strip_height = None  # fixed standard strip height (student); None = the benchmark map's (teacher)
    round_output = False   # round to uint8 instead of truncating (student)
    trt_batch = 1        # strips per TensorRT engine pass: an int, or a function of the padded (height, width)


_runtime = _Runtime()


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


def _postprocess(tensor: torch.Tensor, round_output: bool = False) -> np.ndarray:
    """Clamps to [0,1], converts GPU tensor back to BGR uint8 HWC numpy array. Truncates to uint8 unless round_output."""
    output_tensor = tensor.clamp(0, 1).to(torch.float32)
    scaled = output_tensor.squeeze(0) * 255
    if round_output:
        scaled = scaled.round()
    output_image_converted = scaled.byte().cpu().numpy()
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


def _forward_batch(chunks: list[ChunkMeta], model: torch.nn.Module, scale: int, forward=None) -> list[tuple[np.ndarray, ChunkMeta]]:
    """One forward pass over a batch of same-shape chunks. `forward` replaces the eager model call if given."""
    preprocessed_chunks = []
    for chunk in chunks:
        tensor = _preprocess(chunk.page_slice)
        preprocessed_chunks.append(tensor)

    concated_chunks = torch.cat(preprocessed_chunks)
    chunks_padded, pad_h, pad_w = _pad(concated_chunks, scale)
    with torch.no_grad():
        upscaled = model(chunks_padded) if forward is None else forward(chunks_padded)
    upscaled_unpadded = _unpad(upscaled, pad_h, pad_w, scale)

    individual = torch.split(upscaled_unpadded, 1, dim=0)
    results = []
    for tensor, meta in zip(individual, chunks):
        results.append((_postprocess(tensor, _runtime.round_output), meta))

    return results


def resolve_backend(requested: str, trt_ok: bool | None = None, compile_ok: bool | None = None) -> str:
    """
    Turns the requested backend into the one that can actually run: trt needs tensorrt to import,
    compiled needs torch.compile, and each falls back to the next (trt -> compiled -> eager).
    """
    if requested not in BACKENDS:
        raise ValueError(f"unknown backend {requested!r}, expected one of {BACKENDS}")
    if trt_ok is None:
        trt_ok = trt_backend.is_available()
    if compile_ok is None:
        compile_ok = hasattr(torch, "compile")

    if requested in ("auto", "trt") and trt_ok:
        return "trt"
    if requested in ("auto", "trt", "compiled") and compile_ok:
        return "compiled"
    return "eager"


def select_backend(requested: str, model: torch.nn.Module, model_path: Path, benchmark_map: dict | None,
                   strip_height: int | None = None, round_output: bool = False, trt_batch: int = 1, arch: str = "rrdb") -> str:
    """
    Chooses the runtime for the rest of the run and prints it once. Returns the backend in use.
    strip_height: the model's fixed standard strip height, if it doesn't use the benchmark map's.
    round_output: round the model's output to uint8 instead of truncating it.
    trt_batch: strips per TensorRT engine pass, an int or a function of the padded (height, width).
    arch: short model-family name for the engine file names.
    """
    global _runtime
    _runtime = _Runtime()
    _runtime.failed_shapes = set()
    _runtime.benchmark_map = benchmark_map
    _runtime.strip_height = strip_height
    _runtime.round_output = round_output
    _runtime.trt_batch = trt_batch
    backend = resolve_backend(requested)

    if backend == "trt":
        try:
            _runtime.trt = trt_backend.TrtBackend(model, model_path, arch=arch)
        except Exception as e:
            tqdm.write(f"TensorRT could not start ({type(e).__name__}: {e}), falling back.")
            backend = resolve_backend("compiled")
    if backend == "compiled":
        try:
            _runtime.compiled = torch.compile(copy.deepcopy(model).to(memory_format=torch.channels_last))
        except Exception as e:
            tqdm.write(f"torch.compile could not start ({type(e).__name__}: {e}), falling back.")
            backend = "eager"

    _runtime.backend = backend
    note = "" if requested in (backend, "auto") else f" ({requested} is not available)"
    print(f"Inference backend: {backend}{note}")
    return backend


def is_standard_shape(height: int, width: int, benchmark_map: dict | None, strip_height: int | None = None) -> bool:
    """True if a chunk of this size is its width's standard strip, the only shape worth an engine."""
    if strip_height is not None:
        return height == strip_height
    return height == get_strip_height(width, benchmark_map)


def route(height: int, width: int) -> str:
    """Which runtime a chunk of this size goes to: the selected backend for standard strips, eager for the rest."""
    if _runtime.backend == "eager" or (height, width) in _runtime.failed_shapes:
        return "eager"
    if not is_standard_shape(height, width, _runtime.benchmark_map, _runtime.strip_height):
        return "eager"
    return _runtime.backend


def _padded_shape(height: int, width: int, scale: int = SCALE) -> tuple[int, int]:
    """Shape after _pad: what the engine is built for."""
    return height + (-height) % scale, width + (-width) % scale


def _engine_batch(height: int, width: int) -> int:
    """Batch size of the TensorRT engine for a padded strip shape."""
    batch = _runtime.trt_batch
    return batch(height, width) if callable(batch) else batch


def _fall_back(shape: tuple[int, int], error: Exception) -> None:
    """Logs the failure once and sends this shape to eager for the rest of the run."""
    _runtime.failed_shapes.add(shape)
    tqdm.write(f"{_runtime.backend} backend failed for {shape[0]}x{shape[1]} strips "
               f"({type(error).__name__}: {error}). Using eager for this shape from now on.")


def prepare_chapter(chunks: list[ChunkMeta], scale: int = SCALE) -> None:
    """Builds any TensorRT engine the chapter needs before its first batch, and starts the per-chapter engine stats."""
    if _runtime.trt is None:
        return
    _runtime.trt.switches = 0
    _runtime.trt.load_seconds = []

    for shape in dict.fromkeys(chunk.page_slice.shape[:2] for chunk in chunks):
        if route(*shape) != "trt":
            continue
        height, width = _padded_shape(*shape, scale)
        batch = _engine_batch(height, width)
        if _runtime.trt.has_engine(height, width, batch):
            continue
        try:
            _runtime.trt.build(height, width, batch)
        except Exception as e:
            _fall_back(shape, e)


def engine_stats() -> dict:
    """Engine switches and load times (deserialize + context creation) since the last prepare_chapter()."""
    if _runtime.trt is None:
        return {"switches": 0, "load_seconds": []}
    return {"switches": _runtime.trt.switches, "load_seconds": list(_runtime.trt.load_seconds)}


def _forward_trt(chunks: list[ChunkMeta], scale: int) -> list[tuple[np.ndarray, ChunkMeta]]:
    """
    TensorRT engines have a fixed batch size: runs the chunks through the engine that many at a time.
    A short last group is padded by repeating its last strip, and the padding's output is dropped.
    """
    engine_batch = _engine_batch(*_padded_shape(*chunks[0].page_slice.shape[:2], scale))
    results = []
    for start in range(0, len(chunks), engine_batch):
        group = chunks[start:start + engine_batch]
        tensors = [_preprocess(chunk.page_slice) for chunk in group]
        tensors += [tensors[-1]] * (engine_batch - len(group))
        group_padded, pad_h, pad_w = _pad(torch.cat(tensors), scale)
        upscaled = _unpad(_runtime.trt.run(group_padded), pad_h, pad_w, scale)
        for tensor, chunk in zip(torch.split(upscaled, 1, dim=0), group):
            results.append((_postprocess(tensor, _runtime.round_output), chunk))
    return results


def _forward_compiled(x: torch.Tensor) -> torch.Tensor:
    return _runtime.compiled(x.contiguous(memory_format=torch.channels_last)).contiguous()


def batch_upscale(chunks: list[ChunkMeta], model: torch.nn.Module, scale: int = 4) -> list[tuple[np.ndarray, ChunkMeta]]:
    """
    Upscales a batch of same-shape chunks.
    Standard strips go to the selected backend (see route()); TensorRT takes them in groups of its engine's
    batch size, the others in a single forward pass. Preprocesses each chunk, runs the model, splits and postprocesses each result.
    If the backend raises, the shape falls back to eager for the rest of the run.
    On out-of-memory the batch is split in half and each half retried; a batch of 1 that still fails re-raises.
    Returns paired upscaled arrays and their ChunkMetas.
    """
    shape = chunks[0].page_slice.shape[:2]
    backend = route(*shape)

    if backend == "trt":
        try:
            return _forward_trt(chunks, scale)
        except Exception as e:
            _fall_back(shape, e)
        _runtime.trt.release()
    elif _runtime.trt is not None:
        # One runtime's activation memory at a time: an eager pass must not share VRAM with a live engine.
        _runtime.trt.release()

    forward = _forward_compiled if backend == "compiled" else None
    try:
        return _forward_batch(chunks, model, scale, forward)
    except Exception as e:
        out_of_memory = isinstance(e, RuntimeError) and _is_oom(e)
        if out_of_memory and len(chunks) == 1:
            raise
        if not out_of_memory:
            if forward is None:
                raise
            _fall_back(shape, e)

    # Retry outside the except block, so the failed pass's tensors are no longer held by the traceback.
    torch.cuda.empty_cache()
    if not out_of_memory:
        return batch_upscale(chunks, model, scale)
    mid = len(chunks) // 2
    return batch_upscale(chunks[:mid], model, scale) + batch_upscale(chunks[mid:], model, scale)
