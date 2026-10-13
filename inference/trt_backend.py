"""TensorRT FP16 runtime for the RRDBNet forward pass.

Builds one engine per padded strip shape from an ONNX export of the FP16 model,
caches it in cache/trt_engines/, and runs it on a dedicated CUDA stream. Engines
are batch 1 and only one is kept live at a time (its activation memory is large).

The build and run logic is the one measured in experiments/exp14_tensorrt.py.
tensorrt and onnx are optional: nothing here imports them until an engine is needed.
"""

import gc
import hashlib
import re
import time
from pathlib import Path
import torch
from tqdm import tqdm

ENGINE_DIR = Path("cache/trt_engines")
WORKSPACE_BYTES = 1 << 30
ONNX_OPSET = 17
HASH_LENGTH = 12


def is_available() -> bool:
    """True if tensorrt can be imported."""
    try:
        import tensorrt  # noqa: F401
    except Exception:
        return False
    return True


def _sanitise(text: str) -> str:
    """Keeps letters, digits, dots and hyphens so the text is safe inside a file name."""
    return re.sub(r"[^A-Za-z0-9.]+", "-", text).strip("-")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_key(height: int, width: int, model_name: str, model_hash: str, gpu_name: str, trt_version: str) -> str:
    """
    Engine file name. An engine is only valid for the exact shape, weights, GPU and TensorRT version
    it was built with, so all of them are in the name: a mismatch is a different file, never a reuse.
    """
    return (f"rrdb_{height}x{width}_{_sanitise(model_name)}-{model_hash[:HASH_LENGTH]}"
            f"_{_sanitise(gpu_name).lower()}_trt{_sanitise(trt_version)}_fp16.plan")


class TrtBackend:
    def __init__(self, model: torch.nn.Module, model_path: Path, engine_dir: Path = ENGINE_DIR):
        import tensorrt as trt
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._model = model
        self._model_name = Path(model_path).name
        self._model_hash = file_sha256(Path(model_path))
        self._gpu_name = torch.cuda.get_device_name(0)
        self._engine_dir = Path(engine_dir)
        self._shape = None
        self._engine = None
        self._context = None
        self._stream = None
        self.switches = 0
        self.load_seconds = []

    def engine_path(self, height: int, width: int) -> Path:
        key = cache_key(height, width, self._model_name, self._model_hash, self._gpu_name, self._trt.__version__)
        return self._engine_dir / key

    def has_engine(self, height: int, width: int) -> bool:
        return self.engine_path(height, width).exists()

    def release(self) -> None:
        """Frees the live engine and its activation memory."""
        if self._engine is None:
            return
        self._shape = None
        self._context = None
        self._engine = None
        gc.collect()
        torch.cuda.empty_cache()

    def build(self, height: int, width: int) -> Path:
        """Exports the FP16 model to ONNX for a 1x3xHxW input and builds a strongly typed engine from it."""
        import copy
        trt = self._trt
        plan_path = self.engine_path(height, width)
        tqdm.write(f"building TensorRT engine for {height}x{width}, one-time ~80s")
        self._engine_dir.mkdir(parents=True, exist_ok=True)
        self.release()
        torch.cuda.empty_cache()

        onnx_path = plan_path.with_suffix(".onnx")
        try:
            # FP16 lives in the ONNX file itself: strongly typed networks take their precision from it.
            fp16_model = copy.deepcopy(self._model).half().eval()
            dummy = torch.zeros((1, 3, height, width), device="cuda", dtype=torch.float16)
            with torch.no_grad():
                torch.onnx.export(fp16_model, dummy, str(onnx_path), opset_version=ONNX_OPSET,
                                  input_names=["x"], output_names=["y"], dynamo=False)
            del fp16_model, dummy
            gc.collect()
            torch.cuda.empty_cache()

            builder = trt.Builder(self._logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
            parser = trt.OnnxParser(network, self._logger)
            if not parser.parse(onnx_path.read_bytes()):
                errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                raise RuntimeError(f"ONNX parse failed: {errors}")
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("TensorRT engine build failed")

            # Write to .part and rename, so an interrupted build never leaves a truncated engine behind.
            part_path = plan_path.with_name(plan_path.name + ".part")
            part_path.write_bytes(bytes(serialized))
            part_path.replace(plan_path)
            del builder, network, parser, config, serialized
        finally:
            onnx_path.unlink(missing_ok=True)
            gc.collect()
            torch.cuda.empty_cache()

        return plan_path

    def _activate(self, height: int, width: int) -> None:
        """Makes the engine for this shape the live one, building it first if it isn't cached."""
        if self._shape == (height, width):
            return
        self.release()
        plan_path = self.engine_path(height, width)
        if not plan_path.exists():
            self.build(height, width)

        start = time.perf_counter()
        engine = self._runtime.deserialize_cuda_engine(plan_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"could not load TensorRT engine {plan_path.name}")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError(f"could not create a TensorRT execution context for {plan_path.name}")
        elapsed = time.perf_counter() - start

        self._engine, self._context, self._shape = engine, context, (height, width)
        self.switches += 1
        self.load_seconds.append(elapsed)
        tqdm.write(f"TensorRT engine {height}x{width} loaded in {elapsed:.2f}s")

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """Runs one padded FP16 1x3xHxW tensor through the engine for its shape. Returns the 4x FP16 tensor."""
        trt = self._trt
        batch, _, height, width = x.shape
        if batch != 1:
            raise ValueError(f"TensorRT engines are batch 1, got a batch of {batch}")
        self._activate(height, width)
        engine, context = self._engine, self._context

        x_in = x.half().contiguous()
        input_name, output_name = engine.get_tensor_name(0), engine.get_tensor_name(1)
        output_dtype = torch.float16 if engine.get_tensor_dtype(output_name) == trt.DataType.HALF else torch.float32
        y = torch.empty(tuple(engine.get_tensor_shape(output_name)), dtype=output_dtype, device="cuda")
        context.set_tensor_address(input_name, x_in.data_ptr())
        context.set_tensor_address(output_name, y.data_ptr())

        # TensorRT adds extra syncs on the default stream, so it gets its own.
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        main_stream = torch.cuda.current_stream()
        self._stream.wait_stream(main_stream)
        if not context.execute_async_v3(self._stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        main_stream.wait_stream(self._stream)
        return y.half()
