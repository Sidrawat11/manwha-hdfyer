"""Exp 19 gate: where does the production FP16 TensorRT engine spend its time?

Exports hdfyer to ONNX exactly as hdfyer/digest/trt.py does (FP16 weights, opset 17, TorchScript exporter, static
1x3x512x800: the strip shape Chapter 134 runs at on the laptop), builds a strongly typed engine with detailed profiling
metadata, and times every layer with a TensorRT IProfiler. Layers are grouped by the engine inspector's layer type.

  python experiments/exp19/profile_fp16.py      -> experiments/exp19/fp16_layer_profile.csv (+ engine/ONNX in outputs/exp19-int8/)
"""
import copy, csv, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.getcwd())
import torch
import tensorrt as trt
from tqdm import tqdm
from hdfyer.digest.model import load_student

WEIGHTS = "models/hdfyer/hdfyer-2x-s-r2.pth"
H, W = 512, 800
OUT = "outputs/exp19-int8"
CSV = "experiments/exp19/fp16_layer_profile.csv"
RUNS = 50
LOGGER = trt.Logger(trt.Logger.WARNING)


def export_onnx(path):
    model = copy.deepcopy(load_student(WEIGHTS)).half().eval()
    with torch.no_grad():
        torch.onnx.export(model, torch.zeros((1, 3, H, W), device="cuda", dtype=torch.float16), path, opset_version=17,
                          input_names=["x"], output_names=["y"], dynamo=False)


def build(onnx_path, plan_path):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, LOGGER)
    assert parser.parse(open(onnx_path, "rb").read()), [str(parser.get_error(i)) for i in range(parser.num_errors)]
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    open(plan_path, "wb").write(bytes(builder.build_serialized_network(network, config)))


class LayerTimes(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.ms = defaultdict(float)

    def report_layer_time(self, name, ms):
        self.ms[name] += ms


def category(layer_type, name):
    t, n = (layer_type or "").lower(), name.lower()
    if "conv" in t or "gemm" in t:
        return "convolution"
    if "reformat" in t or "reformat" in n:
        return "reformat"
    if "shuffle" in t or "depthtospace" in n or "pixel" in n:
        return "shuffle / pixel-shuffle"
    if "pointwise" in t or "elementwise" in t or "activation" in t or "sigmoid" in n:
        return "element-wise / sigmoid (fused point-wise)"
    return f"other ({layer_type})"


def main():
    os.makedirs(OUT, exist_ok=True)
    onnx_path, plan_path = f"{OUT}/hdfyer_{H}x{W}_fp16.onnx", f"{OUT}/hdfyer_{H}x{W}_fp16_profiled.plan"
    if not os.path.exists(plan_path):
        print("exporting ONNX and building the FP16 engine (one-time)...", flush=True)
        export_onnx(onnx_path)
        build(onnx_path, plan_path)
    engine = trt.Runtime(LOGGER).deserialize_cuda_engine(open(plan_path, "rb").read())
    info = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    types = {layer["Name"]: layer.get("LayerType", "") for layer in info["Layers"]}

    context = engine.create_execution_context()
    x = torch.rand((1, 3, H, W), device="cuda", dtype=torch.float16)
    y = torch.empty(tuple(engine.get_tensor_shape(engine.get_tensor_name(1))), device="cuda", dtype=torch.float16)
    context.set_tensor_address(engine.get_tensor_name(0), x.data_ptr())
    context.set_tensor_address(engine.get_tensor_name(1), y.data_ptr())
    stream = torch.cuda.Stream()
    for _ in range(5):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # whole-engine time without the profiler, for reference
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
        for _ in tqdm(range(RUNS), desc="engine timing"):
            context.execute_async_v3(stream.cuda_stream)
        end.record()
    stream.synchronize()
    engine_ms = start.elapsed_time(end) / RUNS

    profiler = LayerTimes()
    context.profiler = profiler
    for _ in tqdm(range(RUNS), desc="per-layer profile"):
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    per_layer = {name: ms / RUNS for name, ms in profiler.ms.items()}
    total = sum(per_layer.values())
    groups = defaultdict(lambda: [0.0, 0])
    for name, ms in per_layer.items():
        g = groups[category(types.get(name, ""), name)]
        g[0] += ms; g[1] += 1
    rows = sorted(({"category": c, "layers": n, "ms_per_strip": round(ms, 3), "pct_of_gpu_time": round(100 * ms / total, 1)}
                   for c, (ms, n) in groups.items()), key=lambda r: -r["ms_per_strip"])
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        w.writerow({"category": "total (sum of layers, profiled)", "layers": len(per_layer), "ms_per_strip": round(total, 3), "pct_of_gpu_time": 100.0})
        w.writerow({"category": "whole engine, unprofiled", "layers": "", "ms_per_strip": round(engine_ms, 3), "pct_of_gpu_time": ""})
    print(f"\n{'category':45s} {'layers':>6s} {'ms/strip':>9s} {'% of GPU time':>13s}")
    for r in rows:
        print(f"{r['category']:45s} {r['layers']:6d} {r['ms_per_strip']:9.3f} {r['pct_of_gpu_time']:12.1f}%")
    print(f"{'total (profiled)':45s} {len(per_layer):6d} {total:9.3f}\nwhole engine unprofiled: {engine_ms:.3f} ms/strip")
    print("\nslowest layers:")
    for name, ms in sorted(per_layer.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {ms:7.3f} ms  [{types.get(name, '?')}]  {name[:110]}")
    print("layer types seen:", sorted(set(types.values())))


if __name__ == "__main__":
    main()
