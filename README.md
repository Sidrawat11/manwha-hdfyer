# ManwhaHDFyer

AI-powered manhwa upscaler built for readers who are tired of squinting at compressed webtoon panels. Takes low-quality CBZ chapters downloaded from manga reader apps and produces crisp 2× output using an RRDBNet model (4x-UltraSharp) running on a custom FP16 inference engine.

Built from scratch as a learning project in ML inference, GPU memory profiling, image processing, and software architecture.

---

## Features

- **Custom inference engine.** The RRDBNet forward pass runs directly in PyTorch FP16. Real-ESRGAN is no longer a runtime dependency; only the architecture class comes from `basicsr`.
- **GPU profiler.** On first run it measures peak VRAM and latency for every page shape in your library and caches the results per GPU model.
- **VRAM-aware chunking.** Tall webtoon pages are sliced into fixed-shape strips (about 512 rows for an 800 px wide page), never taller than the GPU can safely hold. The limit comes from a linear VRAM model fitted to profiler data ([below](#the-vram-model)).
- **Batched inference.** Same-shape chunks from any page are stacked into one forward pass, sized by the profiler's safe batch size. This pays off on large GPUs.
- **Seamless stitching.** Strips overlap by about 32 rows and are linearly alpha-blended back together, so there are no seams or doubled panels.
- **Post-processing.** An unsharp mask restores line art, and near-black pixels are clamped to pure black to remove model blotchiness in dark areas.
- **Crash-safe resume.** Chapters are written to `.part` files and renamed only when complete, and finished chapters are skipped on restart.
- **Cloud-ready.** The same code runs on a laptop GPU or a datacenter card; each GPU gets its own profile.

---

## Architecture

```
ManwhaHDFyer/
├── main.py                    ← Entry point: profile (once per GPU), then process the library
├── core/
│   ├── extractor.py           ← CBZ → decoded images
│   ├── model_loader.py        ← Builds RRDBNet, loads weights, FP16 on CUDA
│   ├── packager.py            ← Images → output CBZ (JPEG/PNG)
│   └── config.py              ← Settings dataclasses (not yet wired into the v2 pipeline)
├── profiler/
│   ├── scanner.py             ← Reads page dimensions from CBZ headers, no decoding
│   ├── runner.py              ← Measures VRAM + latency per shape; VRAM estimate formula
│   ├── benchmark.py           ← Benchmark map, batch-size lookup, max chunk height
│   └── cache.py               ← Per-GPU JSON cache in cache/
├── inference/
│   ├── chunker.py             ← Page → fixed-shape overlapping strips (ChunkMeta)
│   ├── batcher.py             ← Groups same-shape chunks into VRAM-safe batches
│   ├── engine.py              ← Pre/post-processing, backend selection, batched forward pass
│   └── trt_backend.py         ← TensorRT FP16 engines: build, cache, run (optional)
├── pipeline/
│   ├── page_buffer.py         ← Collects a page's chunks as they return from the GPU
│   └── reassembler.py         ← Stitches chunks with overlap blending
├── postprocessing/
│   ├── sharpen.py             ← Unsharp mask
│   └── cleanup.py             ← Near-black → black
├── plotter.py                 ← Fits the VRAM model and draws vram_vs_pixels_detailed.png
├── test.py                    ← Chunk/reassemble round-trip test (no inference)
├── experiments/               ← Numbered experiment scripts behind the strip planner
├── legacy/                    ← Phase 1 code, kept for reference (does not run)
├── models/                    ← Weight files (gitignored, download manually)
├── cache/                     ← GPU profiles (gitignored, generated)
├── Manwhas/                   ← Input library (gitignored)
└── results/                   ← Output, mirrors Manwhas/ (gitignored)
```

**Data flow for one chapter:**

```
CBZ → decode pages
    → chunk each page (single pass if it fits, else overlapping strips)
    → group same-shape chunks into batches → FP16 RRDBNet 4× on GPU
    → collect chunks per page → stitch with blended overlaps
    → sharpen → black cleanup → Lanczos downscale to 2×
    → JPEG q92 → output CBZ (.part, renamed on completion)
```

**First run on a new GPU:**

```
scan library headers → unique widths + max height
    → profile each width × {128, 256, 512, 1024, 2048, max} px heights
    → peak VRAM, median latency, safe batch size → cache/<gpu>.json
```

---

## The VRAM model

To pick strip sizes, the chunker needs to know how much VRAM a forward pass will use **before** it runs. An OOM on Windows/WSL doesn't fail cleanly: the driver spills into shared system memory and the run thrashes. So guessing wrong costs minutes, not a quick exception.

The profiler measured peak VRAM for RRDBNet-23 (FP16) across every page width in the library and heights from 128 to 1024 px on an RTX 4060 Laptop (8 GB). Plotting peak VRAM against total pixel count gives an almost perfectly straight line:

![Peak VRAM vs pixel count on RTX 4060 Laptop](vram_vs_pixels_detailed.png)

```
VRAM_MB ≈ 0.0085 × (W × H) + 32.1        R² = 1.0000
```

- **Slope, 0.0085 MB/pixel (≈ 8.5 MB per 1000 px):** activation memory. RRDBNet keeps every intermediate feature map at the input resolution, so memory grows linearly with pixel count.
- **Intercept, 32.1 MB:** fixed cost (model weights in FP16 plus CUDA workspace).
- **Width and height don't matter separately.** Points with different shapes but the same pixel count land on the same line (e.g. 1080×512 and 545×1024), so one variable is enough.

### How the formula is used

| Where | Use |
|---|---|
| `profiler/runner.estimate_shape_vram` | The formula itself. |
| Profiler pre-flight | Shapes estimated above **75%** of VRAM are marked invalid without being run, which avoids the WSL thrash. This is why `709×1024` shows as "OOM" in the 4060 cache. |
| `inference/chunker.chunk_page` | A page under 75% of VRAM is upscaled in one pass. |
| `profiler/benchmark.get_max_chunk_height` | Solves the formula for height: `H = (0.75·VRAM − 32.1) / (0.0085·W)`. On the 4060 that's 898 rows for an 800 px wide page, or 998 for 720 px. This is the ceiling for strip height (and the strip height itself with `LEGACY_STRIPS`). |
| `profiler/benchmark.build_map` | Safe batch size = `floor(0.85·VRAM / peak)`, from measured peaks. |

The 75% budget leaves room for the 4× output tensor and allocator fragmentation; the 85% line caps batch size. Both are drawn on the plot.

### Validation

The current 4060 profile (re-run 2026-05-16) agrees with the formula to within **12 MB** at 35 of 36 measured shapes; refitting those 35 points gives slope 0.00852, intercept 33.2, R² = 0.9999996. The one outlier is `690×1024`, which measured 4456 MB against a predicted ~6040 MB (the original April profile measured ~6050 MB there). The formula is conservative at that point, so it's safe, but the measurement is worth re-checking.

The constants were fitted on the 4060 and are used unchanged on other GPUs. Activation memory per pixel depends on the model, not the card, so they should carry over, but they haven't been re-verified on the H200.

To regenerate the plot from your own profile: `python plotter.py` (reads `cache/rtx-4060-laptop-gpu.json`).

---

## Performance

Measured on an RTX 4060 Laptop (8 GB), 800 px wide pages:

| | |
|---|---|
| One 800×4700 page (GPU) | ~17 s |
| 34-page chapter (wall) | ~10 min |

The GPU is compute-bound: per-pixel cost is roughly flat (2.9–4.3 µs/px) across tile sizes, and only one strip fits per batch, so batching doesn't help on 8 GB cards. It's built for large GPUs; an H200 profile allows batches of up to 225.

### Smarter strips

The strip planner (`inference/chunker.py`) replaced the original one, which cut the tallest strip that fit VRAM (898 rows at 800 px), overlapped strips by 128 rows and left a short last strip. Two things were wasted: the 128-row overlaps are upscaled twice, and the last strip's height changed with every page, so cuDNN re-tuned for a new shape each time (about 7 s the first time a shape appears).

Now every strip of a page has the same height, starts are spread evenly down the page, and neighbours overlap by about 32 rows. Compared with the old planner: ~80% GPU time, ~87% wall time on a chapter, output differences under 1 grey level on average.

| Measurement | Result |
|---|---|
| GPU time, 15 unseen pages (experiment 13) | 80% of the old planner |
| Wall time, one 34-page chapter, full pipeline | 87% (589 s vs 680 s) |
| GPU time, same chapter, raw 2× (validation) | 85% (559 s vs 658 s), min SSIM 0.9990 |
| Similarity to the old output, raw 2× (SSIM, grayscale) | 0.9990–0.9995 per page |
| Mean absolute difference, raw 2× | 0.07 grey levels (of 255) |

Set `LEGACY_STRIPS = True` at the top of `main.py` to get the old planner back. The experiments behind this are in [experiments/](experiments/).

---

## Requirements

- Python 3.10+ (developed on 3.12)
- NVIDIA GPU with CUDA (no CPU fallback)
- 8 GB+ VRAM recommended

---

## Setup

```bash
git clone https://github.com/Sidrawat11/manwha-hdfyer.git
cd manwha-hdfyer
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

**PyTorch with CUDA first:** pip's default index ships CPU-only builds, so install torch from the PyTorch CUDA index (pick your CUDA version on [pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

**Then the rest:**

```bash
pip install -r requirements.txt
```

**`basicsr` fix:** if importing it fails with `No module named 'torchvision.transforms.functional_tensor'`, edit `site-packages/basicsr/data/degradations.py` and change

```python
from torchvision.transforms.functional_tensor import rgb_to_grayscale
```

to

```python
from torchvision.transforms.functional import rgb_to_grayscale
```

**Model weights:** download **4x-UltraSharp.pth** from [OpenModelDB](https://openmodeldb.info/models/4x-UltraSharp) into `models/`.

---

## Usage

Put your library in `Manwhas/` (any folder structure; every `*.cbz` is found recursively), then:

```bash
python main.py
```

On the first run with a new GPU it profiles first. That takes a few minutes and is cached in `cache/<gpu>.json`; delete the file to re-profile. Output goes to `results/`, mirroring the input tree. Interrupted runs resume where they stopped.

Paths and settings are currently hard-coded in `main.py`. The one flag is the inference runtime:

```bash
python main.py --backend auto      # auto (default) | trt | compiled | eager
```

`auto` uses TensorRT if it is installed (`pip install tensorrt-cu12 onnx`, optional, see `requirements.txt`), else `torch.compile`, else plain PyTorch. TensorRT engines are built once per strip shape (about 80 s each) and cached in `cache/trt_engines/`; shapes without an engine, and any engine that fails, run on plain PyTorch.

Finished pages are sharpened, downscaled and JPEG-encoded on background threads while the GPU works on the next pages, instead of the GPU waiting for each one. Three flags control it:

- `--cpu-workers N`: threads doing that work (default `min(4, cpu count)`).
- `--max-inflight N`: most 4× pages queued or in progress at once (default 3); caps memory at about 180 MB per in-flight page.
- `--no-overlap`: the old serial path, for debugging and comparison.

**Test:**

```bash
python test.py [path/to/chapter.cbz]
```

This chunks every page, uses a nearest-neighbour resize as a stand-in for the model, reassembles, and checks the output is pixel-exact, for both the default and the legacy strip planner. It then sweeps synthetic pages of widths 690, 800 and 1080 at every height from 130 to 9000 in steps of 7. No inference runs; it takes about 3 minutes.

---

## How it works

**The model:** 4x-UltraSharp is an RRDBNet (Residual-in-Residual Dense Block Network, 23 blocks) trained with GAN methods. At inference time only the generator runs, a deep CNN that maps low-res pixels to 4× high-res output. The final 2× output comes from a Lanczos downscale of the 4× result, which also suppresses model artifacts.

**Chunking:** manhwa pages are typically 720–800 px wide and 4000–5000 px tall, far too big for one forward pass on consumer GPUs. A page that fits the VRAM budget is upscaled in one pass. Otherwise the chunker cuts it into strips of one fixed height, `512 × 800 / width` rows (512 for an 800 px page, so every width gets the same pixels per strip), capped at the VRAM limit. It uses the fewest strips that give at least 32 rows of overlap and spreads their starts evenly, so the overlaps come out at roughly 32 rows each. A page just under two strips tall gets two slightly taller strips instead, to keep the 32-row overlap. After upscaling, each overlap is linearly blended (top strip fading out, bottom strip fading in), so seams disappear.

**Post-processing:** a gentle unsharp mask (strength 0.3, radius 1.0) recovers line detail the model softens. Pixels whose channels are all below 15 become pure black, which removes patchiness in solid dark regions.

---

## Project history

- **Phase 0: MVP** (`legacy/mvp_upscaler.py`). A 180-line script using Real-ESRGAN: 33 min/chapter, 580 MB of temp files, and a stitching bug that doubled panels.
- **Phase 1: modular rewrite** (`legacy/engine.py`, `legacy/batch.py`). Streaming CBZ → CBZ with no temp files, feathered blending, FP16, sharpening, resume support, and a CLI. Still wrapped Real-ESRGAN, with a fixed 720 px chunk size and aspect-ratio-based chunking.
- **Phase 2: current.** Replaced Real-ESRGAN with a custom engine, added the GPU profiler and VRAM model, VRAM-sized chunking, cross-page batching, and the page buffer/reassembler.

---

## Roadmap

### Done
- [x] MVP upscaler
- [x] Modular architecture
- [x] Feathered blending, FP16 inference, sharpening + black cleanup
- [x] Custom inference engine (Real-ESRGAN removed)
- [x] GPU profiler + VRAM model + per-GPU cache
- [x] VRAM-aware chunking and batched inference
- [x] Crash-safe resume
- [x] Fixed-shape strips with 32-row overlap (one cuDNN tune per page width)

### Next
- [ ] CLI wired to `core/config.py` (paths, model, limit, format/quality)
- [ ] Restore credit-page skip and file logging from Phase 1
- [x] `requirements.txt`
- [ ] Overlap CPU work (decode, sharpen, encode) with GPU inference
- [ ] Full library run on H200
- [ ] Edge-aware sharpening (sharpen lines, preserve flat areas)
- [ ] pytest suite (lookup interpolation, chunker edge cases, packager)

### Future
- [ ] Model selection (e.g. RealESRGAN anime_6B)
- [ ] Desktop GUI comparer (Tauri + React): slider, synced zoom/pan, drag-and-drop
- [ ] CPU inference fallback
- [ ] Reverse proxy for real-time Mihon integration (Suwayomi-based)

---

## License

MIT
