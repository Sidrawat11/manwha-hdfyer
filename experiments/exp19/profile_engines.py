"""Exp 19: per-layer time of the FP16, INT8-default and INT8-mixed engines at 512x800, grouped by what the layer does
(read from the layer names TensorRT reports). Shows where INT8 gains (the convolutions) and loses (fusions broken by
the Q/DQ nodes: standalone element-wise layers, extra reformats, standalone quantize/dequantize).

  python experiments/exp19/profile_engines.py   -> experiments/exp19/engine_layer_profile.csv
"""
import csv, os, sys
from collections import defaultdict
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "exp19"))
import torch, tensorrt as trt
from tqdm import tqdm
import engines

RUNS = 30


class LayerTimes(trt.IProfiler):
    def __init__(self):
        super().__init__(); self.ms = defaultdict(float)

    def report_layer_time(self, name, ms):
        self.ms[name] += ms


def category(name):
    n = name.lower()
    if "reformat" in n or "copy" in n:
        return "reformat / copy"
    if "quantize" in n and "conv" not in n:
        return "standalone quantize / dequantize"
    if "conv" in n:
        return "convolution (with whatever TensorRT fused into it)"
    if "depthtospace" in n or "shuffle" in n:
        return "shuffle / pixel-shuffle"
    if any(k in n for k in ("sigmoid", "mul", "add", "sub", "pwn")):
        return "standalone element-wise"
    return "other"


def main():
    rows = []
    for precision in tqdm(["fp16", "int8-default", "int8-mixed"], desc="engines"):
        path = engines.build_fp16(512, 800) if precision == "fp16" else engines.build_int8(precision, 512, 800)[0]
        eng = trt.Runtime(engines.LOGGER).deserialize_cuda_engine(open(path, "rb").read())
        ctx = eng.create_execution_context()
        dtype = lambda i: torch.float16 if eng.get_tensor_dtype(eng.get_tensor_name(i)) == trt.DataType.HALF else torch.float32
        x = torch.rand((1, 3, 512, 800), device="cuda", dtype=dtype(0))
        y = torch.empty(tuple(ctx.get_tensor_shape(eng.get_tensor_name(1))), device="cuda", dtype=dtype(1))
        ctx.set_tensor_address(eng.get_tensor_name(0), x.data_ptr()); ctx.set_tensor_address(eng.get_tensor_name(1), y.data_ptr())
        stream = torch.cuda.Stream()
        for _ in range(5):
            ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        profiler = LayerTimes(); ctx.profiler = profiler
        for _ in range(RUNS):
            ctx.execute_async_v3(stream.cuda_stream); stream.synchronize()
        total = sum(profiler.ms.values()) / RUNS
        groups = defaultdict(lambda: [0.0, 0])
        for name, ms in profiler.ms.items():
            g = groups[category(name)]; g[0] += ms / RUNS; g[1] += 1
        for cat, (ms, n) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
            rows.append({"engine": precision, "category": cat, "layers": n, "ms_per_strip": round(ms, 3), "pct": round(100 * ms / total, 1)})
        rows.append({"engine": precision, "category": "total (profiled)", "layers": len(profiler.ms), "ms_per_strip": round(total, 3), "pct": 100.0})
    with open("experiments/exp19/engine_layer_profile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['engine']:13s} {r['category']:52s} {r['layers']:3d} {r['ms_per_strip']:7.3f} ms {r['pct']:5.1f}%")


if __name__ == "__main__":
    main()
