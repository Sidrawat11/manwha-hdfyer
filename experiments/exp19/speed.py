"""Exp 19 step 4: speed on the laptop, one 34-page 800 px chapter, FP16 vs INT8 default vs INT8 mixed.

Each run is a fresh process running the real pipeline (hdfyer.pipeline.process_all, batch 1, TensorRT) with production's
engine lookup pointed at the variant's engine file; then it times the engine alone. Runs are interleaved
fp16 / int8-default / int8-mixed, 4 rounds, with a 60 s cool-down before each, and the GPU's clock-event reasons
(nvidia-smi -q -d PERFORMANCE) are logged around every run.

  python experiments/exp19/speed.py   -> experiments/exp19/speed_runs.csv
"""
import csv, os, re, shutil, subprocess, sys, time
from pathlib import Path
from tqdm import tqdm

import glob
SPEED_CHAPTER = sorted(glob.glob("bench_chapters/*/*Chapter 134_*.cbz"))   # the 34-page, 800 px chapter used throughout
OUT = Path("experiments/out/exp19/speed"); CSV = "experiments/exp19/speed_runs.csv"
PRECISIONS, ROUNDS, COOL_S, PAGES = ["fp16", "int8-default", "int8-mixed"], 4, 60, 34
RUN = r'''
import os, sys, time, re
sys.path.insert(0, "."); sys.path.insert(0, "experiments/exp19")
from pathlib import Path
import engines
from hdfyer.digest import trt as prod_trt
precision = {precision!r}
if precision != "fp16":
    import torch, tensorrt as trt
    def engine_path(self, h, w, batch=1):
        assert batch == 1
        return Path(engines.build_int8(precision, h, w)[0])
    def run(self, x):
        # production's run, except that the input is cast to what the engine takes: the INT8 (Q/DQ) engines take FP32
        batch, _, h, w = x.shape
        self._activate(h, w, batch)
        eng, ctx = self._engine, self._context
        in_name, out_name = eng.get_tensor_name(0), eng.get_tensor_name(1)
        x_in = (x.half() if eng.get_tensor_dtype(in_name) == trt.DataType.HALF else x.float()).contiguous()
        out_dtype = torch.float16 if eng.get_tensor_dtype(out_name) == trt.DataType.HALF else torch.float32
        y = torch.empty(tuple(ctx.get_tensor_shape(out_name)), dtype=out_dtype, device="cuda")
        ctx.set_tensor_address(in_name, x_in.data_ptr()); ctx.set_tensor_address(out_name, y.data_ptr())
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        main = torch.cuda.current_stream(); self._stream.wait_stream(main)
        assert ctx.execute_async_v3(self._stream.cuda_stream)
        main.wait_stream(self._stream)
        return y.half()
    prod_trt.TrtBackend.engine_path = engine_path
    prod_trt.TrtBackend.run = run
from hdfyer.pipeline import process_all
process_all(Path({lib!r}), Path({out!r}), batch=1)
# engine alone, at the chapter's strip shape
import torch, tensorrt as trt
path = engines.build_fp16(512, 800) if precision == "fp16" else engines.build_int8(precision, 512, 800)[0]
eng = trt.Runtime(engines.LOGGER).deserialize_cuda_engine(open(path, "rb").read()); ctx = eng.create_execution_context()
dt = torch.float16 if eng.get_tensor_dtype(eng.get_tensor_name(0)) == trt.DataType.HALF else torch.float32
x = torch.rand((1, 3, 512, 800), device="cuda", dtype=dt)
y = torch.empty(tuple(ctx.get_tensor_shape(eng.get_tensor_name(1))), device="cuda", dtype=torch.float16 if eng.get_tensor_dtype(eng.get_tensor_name(1)) == trt.DataType.HALF else torch.float32)
ctx.set_tensor_address(eng.get_tensor_name(0), x.data_ptr()); ctx.set_tensor_address(eng.get_tensor_name(1), y.data_ptr())
s = torch.cuda.Stream()
for _ in range(10): ctx.execute_async_v3(s.cuda_stream)
s.synchronize(); a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
with torch.cuda.stream(s):
    a.record()
    for _ in range(200): ctx.execute_async_v3(s.cuda_stream)
    b.record()
s.synchronize(); print(f"ENGINE_MS {{a.elapsed_time(b) / 200:.3f}}")
'''


def clock_events():
    q = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE"], capture_output=True, text=True).stdout
    counters = {k: int(v) for k, v in re.findall(r"(SW Power Capping|SW Thermal Slowdown|HW Thermal Slowdown|HW Power Braking)\s*:\s*(\d+) us", q)}
    active = sorted(set(re.findall(r"^\s+([A-Za-z ]+?)\s*:\s*Active$", q, re.M)))
    temp = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return counters, active, temp


def main():
    assert len(SPEED_CHAPTER) == 1, SPEED_CHAPTER
    lib = OUT / "library" / "chapter"; lib.mkdir(parents=True, exist_ok=True)
    link = lib / "speed_chapter.cbz"
    if not link.exists():
        os.symlink(os.path.abspath(SPEED_CHAPTER[0]), link)
    order = [p for _ in range(ROUNDS) for p in PRECISIONS]
    rows = []
    for i, precision in enumerate(tqdm(order, desc="runs")):
        time.sleep(COOL_S)
        res = OUT / f"results_{i}"; shutil.rmtree(res, ignore_errors=True)
        before, _, temp_start = clock_events()
        t = time.perf_counter()
        p = subprocess.run([sys.executable, "-c", RUN.format(precision=precision, lib=str(OUT / "library"), out=str(res))],
                           capture_output=True, text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        wall = time.perf_counter() - t
        after, active, temp_end = clock_events()
        (OUT / f"run_{i}_{precision}.log").write_text(p.stdout + p.stderr)
        m = re.search(r"Done: .+? in ([\d.]+)s", p.stdout); e = re.search(r"ENGINE_MS ([\d.]+)", p.stdout)
        row = {"run": i, "round": i // len(PRECISIONS) + 1, "precision": precision, "ok": p.returncode == 0 and "Fallbacks: none" in p.stdout,
               "chapter_s": float(m.group(1)) if m else None, "s_per_page": round(float(m.group(1)) / PAGES, 4) if m else None,
               "engine_ms_per_strip": float(e.group(1)) if e else None, "process_s": round(wall, 1),
               "gpu_temp_start_c": temp_start, "gpu_temp_end_c": temp_end,
               **{f"{k.lower().replace(' ', '_')}_us": after.get(k, 0) - before.get(k, 0) for k in after},
               "clock_events_active_after": ";".join(active)}
        rows.append(row); tqdm.write(str(row))
        shutil.rmtree(res, ignore_errors=True)
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for p in PRECISIONS:
        R = [r for r in rows if r["precision"] == p and r["ok"]]
        if R:
            print(f"{p:13s} s/page {[r['s_per_page'] for r in R]}  engine ms/strip {[r['engine_ms_per_strip'] for r in R]}")


if __name__ == "__main__":
    main()
