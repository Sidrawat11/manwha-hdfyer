# Experiments

Experiments 01-15 ran against the pre-restructure layout: `git checkout pre-restructure` to rerun them.

Each experiment is numbered to match the lab notebook (the shared doc). All use the same
60-page sample (`out/sample.json`, seed 2026: 20 random chapters per series, one random page each).
Run everything from the repo root, e.g. `python experiments/exp02_tile_census.py`.

| # | Script | Runs on | Question |
|---|---|---|---|
| 1 | exp01_tapestry.py | CPU | Can a page be split into lines + tapestry and put back without damage? |
| 2 | exp02_tile_census.py | CPU | How much of a page has lines / shading / flat colour? |
| 3 | exp03_margin_cost_model.py | CPU | How much work do context margins add? |
| 4 | exp04_gutters.py | CPU | How much of a page is full-width blank gap? |
| 5 | exp05_panel_crop.py | CPU | Does boxing each panel skip more than gutters? |
| 6 | exp06_split_vs_full.py | GPU | Is edges-only faster than the full pipeline, at what quality? |
| 7 | (diagnosis, images in out/exp06_split_vs_full/fail_*) | CPU | Why did 4 pages fail? |
| 8 | exp08_detail_detector.py | CPU | Does a detail-based detector catch what the line detector missed? |
| 9 | exp09_detail_gpu.py | GPU | Does the detail detector fix the failing pages on the real network, at what speed? |
| 10 | exp10_detail_confirm.py | GPU | Does the detail detector pass the bar on all 60 random pages? |
| 11 | exp11_hybrid_router.py | CPU | Does a per-page full-vs-route switch beat always routing? |
| 12 | exp12_strip_sweep.py | GPU | Are the full pipeline strips too tall, overlaps too big? (15 fresh pages, saves the art) |
| 13 | exp13_strips_vs_routing.py | GPU | Smarter strips vs edge routing head to head (15 new pages) + seam check |
| 14 | exp14_tensorrt.py | GPU | Same model, three runtimes: eager vs torch.compile vs TensorRT FP16 (10 new pages, 4 widths, slider comparers). Needs `pip install onnx tensorrt-cu12` |
| 15 | exp15_native2x.py | GPU | Native 2x models (IllustrationJaNai FDAT/SPAN/DeJPEG, HFA2kCompact) vs today's 4x-then-halve, on the Exp 14 pages. Needs `pip install spandrel` |

`common.py` holds the shared sampling, line detector and tile classifier.
Outputs go to `out/` (gitignored).
