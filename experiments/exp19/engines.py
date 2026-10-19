"""Exp 19: build and load TensorRT engines for one precision at one strip shape, cached under outputs/exp19-int8/engines/.

The file name is production's cache key with the precision added (..._fp16.plan becomes ..._int8-default.plan), so
an INT8 engine can never be mistaken for an FP16 one. The FP16 reference is production itself: its engine comes from
cache/trt_engines/ through hdfyer.digest.trt (built there by production code if missing).
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
import tensorrt as trt
from hdfyer.digest import trt as prod_trt

WEIGHTS = "models/hdfyer/hdfyer-2x-s-r2.pth"
OUT = "outputs/exp19-int8"
ENGINE_DIR = f"{OUT}/engines"
LOGGER = trt.Logger(trt.Logger.WARNING)
_hash = None


def engine_path(precision: str, height: int, width: int) -> str:
    """production cache key + precision. FP16 is production's own file."""
    global _hash
    if _hash is None:
        _hash = prod_trt.file_sha256(WEIGHTS)
    import torch
    key = prod_trt.cache_key(height, width, os.path.basename(WEIGHTS), _hash, torch.cuda.get_device_name(0), trt.__version__, 1, "span")
    if precision == "fp16":
        return os.path.join(str(prod_trt.ENGINE_DIR), key)
    return os.path.join(ENGINE_DIR, key.replace("_fp16.plan", f"_{precision}.plan"))


def build_int8(precision: str, height: int, width: int) -> tuple[str, float]:
    """Builds (once) a strongly typed engine from the Q/DQ ONNX, at a fixed 1x3xHxW. Returns (path, build seconds)."""
    path = engine_path(precision, height, width)
    if os.path.exists(path):
        return path, 0.0
    os.makedirs(ENGINE_DIR, exist_ok=True)
    start = time.perf_counter()
    builder = trt.Builder(LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, LOGGER)
    onnx_path = f"{OUT}/hdfyer_{precision}.onnx"
    if not parser.parse(open(onnx_path, "rb").read()):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    profile = builder.create_optimization_profile()
    shape = (1, 3, height, width)
    profile.set_shape(network.get_input(0).name, shape, shape, shape)
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"{precision} engine build failed for {height}x{width}")
    open(path + f".{os.getpid()}.part", "wb").write(bytes(serialized))
    os.replace(path + f".{os.getpid()}.part", path)
    return path, time.perf_counter() - start


def build_fp16(height: int, width: int) -> str:
    """Production's FP16 engine for this shape, built by production code if it isn't cached yet."""
    path = engine_path("fp16", height, width)
    if not os.path.exists(path):
        from hdfyer.digest.model import load_student
        backend = prod_trt.TrtBackend(load_student(WEIGHTS), WEIGHTS, arch="span")
        backend.build(height, width, 1)
    return path
