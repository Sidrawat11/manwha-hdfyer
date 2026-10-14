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

Every departure from the selected backend is a fallback event: it is logged when it
happens (which pages, which strip shape, why) and kept for the end-of-run summary
(fallback_events). Out-of-memory is retried with a smaller batch after the failed
engine and buffers have been freed.
"""

import copy
import gc
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from inference import trt_backend
from inference.chunker import ChunkMeta, crop_upscaled, get_strip_height

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
    batch_override = {}  # padded (h, w) -> engine batch, lowered after an out-of-memory
    events = []          # fallback events, see fallback_events()
    context = ""         # what is being processed (the chapter), for the fallback log


_runtime = _Runtime()


def _preprocess(image: np.ndarray) -> torch.Tensor:
    """
    Converts a BGR uint8 HWC numpy array to an RGB float16 1x3xHxW tensor on the GPU, values in [0, 1].
    Uploads the uint8 bytes and converts on the GPU: a quarter of the data a float32 upload moves, and no CPU work.
    The result is bit-identical to converting on the CPU (float32 / 255, then fp16) for every possible input value.
    """
    host = torch.from_numpy(np.ascontiguousarray(image))
    rgb = host.to(DEVICE).permute(2, 0, 1).flip(0)          # HWC BGR -> CHW RGB
    return (rgb.unsqueeze(0).float() / 255.0).half()


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


def _to_bgr_uint8(tensor: torch.Tensor, round_output: bool = False) -> torch.Tensor:
    """Clamps a 1x3xHxW tensor to [0,1] and converts it, where it is, to BGR uint8 HxWx3. Truncates to uint8 unless round_output."""
    output_tensor = tensor.clamp(0, 1).to(torch.float32)
    scaled = output_tensor.squeeze(0) * 255
    if round_output:
        scaled = scaled.round()
    return scaled.byte().flip(0).permute(1, 2, 0)   # CHW RGB -> HWC BGR (a view; .contiguous() makes it real)


def _postprocess(tensor: torch.Tensor, round_output: bool = False) -> np.ndarray:
    """
    Clamps to [0,1] and converts a 1x3xHxW GPU tensor to a BGR uint8 HWC numpy array. Truncates to uint8 unless round_output.
    Everything but the final copy runs on the GPU, so the CPU receives finished bytes in the layout OpenCV uses;
    transposing and swapping channels on the CPU cost more than the model's forward pass for a 2x strip.
    """
    return _to_bgr_uint8(tensor, round_output).contiguous().cpu().numpy()


def upscale(image: np.ndarray, model: torch.nn.Module, scale: int = 4) -> np.ndarray:
    """Upscales a single BGR uint8 image. Calls preprocess, pad, model forward, unpad, postprocess."""
    x = _preprocess(image)
    x_padded, pad_h, pad_w = _pad(x, scale)
    with torch.no_grad():
        y = model(x_padded)
    y_unpadded = _unpad(y, pad_h, pad_w, scale)
    img = _postprocess(y_unpadded)
    return img


def _is_oom(error: Exception) -> bool:
    """
    True for PyTorch's OOM and for the driver's. With NVIDIA's "Prefer No Sysmem Fallback" it arrives as a plain CUDA RuntimeError.
    Also true for the allocator's internal assert, which is what the first allocation after a driver OOM raises
    (with expandable segments) if the cache was not emptied in between.
    """
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error)
    return "out of memory" in message.lower() or ("INTERNAL ASSERT FAILED" in message and "CUDACachingAllocator" in message)


def _describe(error: Exception) -> str:
    text = str(error).strip()
    return f"{type(error).__name__}: {text.splitlines()[0]}" if text else type(error).__name__


def adopt_warm_up(failed_shapes, batch_override: dict) -> None:
    """Starts this process from what a warm-up found: shapes that must run eager and engine batches it had to lower."""
    _runtime.failed_shapes = {tuple(shape) for shape in failed_shapes}
    _runtime.batch_override = {tuple(shape): batch for shape, batch in batch_override}


def warm_up_state() -> dict:
    """What adopt_warm_up needs, in a form that pickles."""
    return {"failed_shapes": sorted(_runtime.failed_shapes), "batch_override": sorted(_runtime.batch_override.items())}


def free_gpu() -> None:
    """Frees the live engine and every cached block. After an out-of-memory this must happen before the next allocation."""
    if _runtime.trt is not None:
        _runtime.trt.release()
    gc.collect()
    torch.cuda.empty_cache()


def set_context(context: str) -> None:
    """Names what is being processed (e.g. the chapter), so fallback events can say where they happened."""
    _runtime.context = context


def fallback_events() -> list[dict]:
    """Every fallback since select_backend(): kind, context, pages, shape, reason."""
    return list(_runtime.events)


def _record(kind: str, reason: str, shape: tuple[int, int] | None = None, chunks: list[ChunkMeta] | None = None) -> None:
    """Logs a fallback loudly, now, and keeps it for the end-of-run summary."""
    pages = list(dict.fromkeys(chunk.page_name or f"page {chunk.page_id}" for chunk in chunks or []))
    event = {"kind": kind, "context": _runtime.context, "pages": pages,
             "shape": f"{shape[0]}x{shape[1]}" if shape else "", "reason": reason}
    _runtime.events.append(event)
    tqdm.write("!!! FALLBACK: " + format_event(event))


def report_fallback(kind: str, reason: str) -> None:
    """For callers that fall back before any strip exists (e.g. choosing the backend): logged and summarised like the rest."""
    _record(kind, reason)


def format_event(event: dict) -> str:
    where = [event["context"]] if event["context"] else []
    if event["pages"]:
        where.append("pages " + ", ".join(event["pages"]))
    if event["shape"]:
        where.append(f"strips {event['shape']}")
    return f"[{event['kind']}] " + (" | ".join(where) + ": " if where else "") + event["reason"]


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
        results.append((crop_upscaled(_postprocess(tensor, _runtime.round_output), meta, scale), meta))

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
    _runtime.batch_override = {}
    _runtime.events = []
    backend = resolve_backend(requested)
    if requested != "auto" and backend != requested:
        _record("backend unavailable", f"{requested} was requested but cannot run here; using {backend}")

    if backend == "trt":
        try:
            _runtime.trt = trt_backend.TrtBackend(model, model_path, arch=arch)
        except Exception as e:
            backend = resolve_backend("compiled")
            _record("backend unavailable", f"TensorRT could not start ({_describe(e)}); using {backend}")
    if backend == "compiled":
        try:
            _runtime.compiled = torch.compile(copy.deepcopy(model).to(memory_format=torch.channels_last))
        except Exception as e:
            backend = "eager"
            _record("backend unavailable", f"torch.compile could not start ({_describe(e)}); using eager")

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
    """Batch size of the TensorRT engine for a padded strip shape: the configured one, unless an out-of-memory lowered it."""
    if (height, width) in _runtime.batch_override:
        return _runtime.batch_override[(height, width)]
    batch = _runtime.trt_batch
    return batch(height, width) if callable(batch) else batch


def _fall_back(shape: tuple[int, int], reason: str, chunks: list[ChunkMeta] | None = None) -> None:
    """Sends this shape to eager for the rest of the run, and records why."""
    _runtime.failed_shapes.add(shape)
    _record(f"{_runtime.backend} -> eager", f"{reason}. Eager for this shape for the rest of the run", shape, chunks)


def _shrink_or_fall_back(shape: tuple[int, int], scale: int, reason: str, shrinkable: bool, chunks: list[ChunkMeta] | None = None) -> None:
    """
    After a TensorRT failure for a shape: frees the GPU, then either halves the engine batch (out-of-memory
    or a failed build, while the batch is above 1) or gives the shape to eager.
    """
    free_gpu()
    padded = _padded_shape(*shape, scale)
    batch = _engine_batch(*padded)
    if shrinkable and batch > 1:
        _runtime.batch_override[padded] = batch // 2
        _record("smaller batch", f"{reason}. Retrying with a TensorRT batch of {batch // 2} instead of {batch}", shape, chunks)
    else:
        _fall_back(shape, reason, chunks)


def prepare_chapter(chunks: list[ChunkMeta], scale: int = SCALE) -> None:
    """Builds any TensorRT engine the chapter needs before its first batch, and starts the per-chapter engine stats."""
    if _runtime.trt is None:
        return
    _runtime.trt.switches = 0
    _runtime.trt.load_seconds = []
    if not getattr(_runtime.trt, "allow_build", True):
        return   # a --parallel worker: engines were built in the warm-up; a missing one surfaces (loudly) when used

    by_shape = {}
    for chunk in chunks:
        by_shape.setdefault(chunk.page_slice.shape[:2], []).append(chunk)
    for shape, shape_chunks in by_shape.items():
        # A failed build is retried with half the batch until it works or the shape goes to eager.
        while route(*shape) == "trt":
            height, width = _padded_shape(*shape, scale)
            batch = _engine_batch(height, width)
            if _runtime.trt.has_engine(height, width, batch):
                break
            try:
                _runtime.trt.build(height, width, batch)
                break
            except Exception as e:
                reason = _describe(e)
            _shrink_or_fall_back(shape, scale, reason, True, shape_chunks)


def engine_stats() -> dict:
    """Engine switches and load times (deserialize + context creation) since the last prepare_chapter()."""
    if _runtime.trt is None:
        return {"switches": 0, "load_seconds": [], "build_seconds": []}
    return {"switches": _runtime.trt.switches, "load_seconds": list(_runtime.trt.load_seconds),
            "build_seconds": list(_runtime.trt.build_seconds)}


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
            results.append((crop_upscaled(_postprocess(tensor, _runtime.round_output), chunk, scale), chunk))
    return results


def _forward_compiled(x: torch.Tensor) -> torch.Tensor:
    return _runtime.compiled(x.contiguous(memory_format=torch.channels_last)).contiguous()


def batch_upscale(chunks: list[ChunkMeta], model: torch.nn.Module, scale: int = 4, _last_chance: bool = False) -> list[tuple[np.ndarray, ChunkMeta]]:
    """
    Upscales a batch of same-shape chunks.
    Standard strips go to the selected backend (see route()); TensorRT takes them in groups of its engine's
    batch size, the others in a single forward pass. Preprocesses each chunk, runs the model, splits and postprocesses each result.
    Any departure from the selected backend is recorded as a fallback event (see fallback_events()).
    If the backend raises, the shape falls back to eager for the rest of the run.
    On out-of-memory the failed engine and buffers are freed and the work is retried smaller: TensorRT with half
    the engine batch, eager with the batch split in half. A batch of 1 is retried once on a clean cache, then re-raises.
    Returns paired upscaled arrays and their ChunkMetas.
    """
    shape = chunks[0].page_slice.shape[:2]
    backend = route(*shape)

    if backend == "trt":
        try:
            return _forward_trt(chunks, scale)
        except Exception as e:
            reason = _describe(e)
            # A missing engine in a process that may not build one goes straight to eager: a smaller batch won't exist either.
            shrinkable = (_is_oom(e) or isinstance(e, trt_backend.EngineBuildError)) and not isinstance(e, trt_backend.EngineMissingError)
        # Outside the except block, so the failed pass's tensors are no longer held by the traceback.
        _shrink_or_fall_back(shape, scale, reason, shrinkable, chunks)
        return batch_upscale(chunks, model, scale)

    if backend == "eager" and _runtime.backend != "eager" and shape not in _runtime.failed_shapes:
        _record(f"{_runtime.backend} -> eager", "not the standard strip shape, so there is no engine for it", shape, chunks)
    if _runtime.trt is not None:
        # One runtime's activation memory at a time: an eager pass must not share VRAM with a live engine.
        _runtime.trt.release()

    forward = _forward_compiled if backend == "compiled" else None
    try:
        return _forward_batch(chunks, model, scale, forward)
    except Exception as e:
        out_of_memory = _is_oom(e)
        reason = _describe(e)
        if out_of_memory and len(chunks) == 1 and _last_chance:
            raise
        if not out_of_memory and forward is None:
            raise

    # Retry outside the except block, so the failed pass's tensors are no longer held by the traceback.
    free_gpu()
    if not out_of_memory:
        _fall_back(shape, reason, chunks)
        return batch_upscale(chunks, model, scale)
    if len(chunks) == 1:
        _record("out of memory", f"{reason}. Retrying this strip once on an emptied cache", shape, chunks)
        return batch_upscale(chunks, model, scale, _last_chance=True)
    mid = len(chunks) // 2
    _record("out of memory", f"{reason}. Retrying {len(chunks)} strips as batches of {mid} and {len(chunks) - mid}", shape, chunks)
    return batch_upscale(chunks[:mid], model, scale) + batch_upscale(chunks[mid:], model, scale)


# ---------------------------------------------------------------- one batch in flight

_copy_stream = None   # device-to-host copies run here, beside the next batch's compute


def _launch(chunks: list[ChunkMeta], model: torch.nn.Module, scale: int):
    """
    Queues a batch on the GPU (upload, forward pass, output conversion and crop) and starts copying its results
    into pinned host memory on a separate stream, without waiting for any of it. Returns a handle for _collect,
    or None if the batch should take the synchronous path: a shape that isn't the backend's, or anything that raises
    here (out-of-memory, a missing engine, ...), whose recovery batch_upscale owns.
    """
    global _copy_stream
    shape = chunks[0].page_slice.shape[:2]
    backend = route(*shape)
    if backend != _runtime.backend:
        return None
    try:
        outputs = []
        if backend == "trt":
            engine_batch = _engine_batch(*_padded_shape(*shape, scale))
            for start in range(0, len(chunks), engine_batch):
                group = chunks[start:start + engine_batch]
                tensors = [_preprocess(chunk.page_slice) for chunk in group]
                tensors += [tensors[-1]] * (engine_batch - len(group))
                group_padded, pad_h, pad_w = _pad(torch.cat(tensors), scale)
                upscaled = _unpad(_runtime.trt.run(group_padded), pad_h, pad_w, scale)
                outputs += [crop_upscaled(_to_bgr_uint8(t, _runtime.round_output), c, scale).contiguous()
                            for t, c in zip(torch.split(upscaled, 1, dim=0), group)]
        else:
            if _runtime.trt is not None:
                _runtime.trt.release()
            batch = torch.cat([_preprocess(chunk.page_slice) for chunk in chunks])
            batch_padded, pad_h, pad_w = _pad(batch, scale)
            with torch.no_grad():
                upscaled = model(batch_padded) if backend == "eager" else _forward_compiled(batch_padded)
            upscaled = _unpad(upscaled, pad_h, pad_w, scale)
            outputs = [crop_upscaled(_to_bgr_uint8(t, _runtime.round_output), c, scale).contiguous()
                       for t, c in zip(torch.split(upscaled, 1, dim=0), chunks)]

        if _copy_stream is None:
            _copy_stream = torch.cuda.Stream()
        computed = torch.cuda.Event()
        computed.record()
        _copy_stream.wait_event(computed)
        with torch.cuda.stream(_copy_stream):
            hosts = [torch.empty(t.shape, dtype=torch.uint8, pin_memory=True).copy_(t, non_blocking=True) for t in outputs]
            copied = torch.cuda.Event()
            copied.record()
        for t in outputs:
            t.record_stream(_copy_stream)   # the allocator must not reuse them until the copy has read them
        return hosts, copied, outputs
    except Exception:
        return None


def _collect(chunks: list[ChunkMeta], handle, model: torch.nn.Module, scale: int) -> list[tuple[np.ndarray, ChunkMeta]]:
    """Waits for a launched batch's copy to finish and returns its results; a batch that wasn't launched runs now."""
    if handle is None:
        return batch_upscale(chunks, model, scale)
    hosts, copied, _ = handle
    copied.synchronize()
    # numpy views of the pinned tensors: each view keeps its tensor (and the pinned block) alive while it is used.
    return [(host.numpy(), chunk) for host, chunk in zip(hosts, chunks)]


def upscale_stream(batches: list[list[ChunkMeta]], model: torch.nn.Module, scale: int = 4):
    """
    Yields (batch, results) for each batch, in order, keeping one batch in flight: batch k+1 is uploaded and queued on
    the GPU, and batch k's results are already copying back, before the caller gets batch k. So the caller's work on a
    batch (stitching, handing pages to the finisher) overlaps the GPU's work on the next one.
    Results are identical to calling batch_upscale on each batch, which is what happens on the CPU and for any batch
    that needs the fallback or out-of-memory handling.
    """
    if DEVICE.type != "cuda":
        for batch in batches:
            yield batch, batch_upscale(batch, model, scale)
        return
    pending = None
    for batch in batches:
        handle = _launch(batch, model, scale)
        if pending is not None:
            yield pending[0], _collect(pending[0], pending[1], model, scale)
        pending = (batch, handle)
    if pending is not None:
        yield pending[0], _collect(pending[0], pending[1], model, scale)
