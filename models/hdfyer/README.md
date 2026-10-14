# HDFyer 2x models

Our own 2x upscalers, distilled from the production look (4x-UltraSharp, halved, before sharpening).
Weights are not committed (*.pth is gitignored); this card is.

Naming: `hdfyer-2x-<size>-r<run>.pth`
  2x    native output scale
  s     student size (s = SPAN S, 48 channels)
  r2    training run number in experiments/out/exp16_cloud/ and experiments/exp16/runs.csv

## hdfyer-2x-s-r2.pth  (current)
- Architecture: SPAN S, 2x (load with spandrel, as experiments/exp16/distill.py load_model)
- Initialised from: 2x_IllustrationJaNai_V3detail_SPAN_S_40k (CC BY-NC 4.0)
- Teacher: 4x-UltraSharp, strips stitched, Lanczos-halved to 2x, no sharpening
- Training: 548 pages from 3 series, 60k steps, L1 + edge loss 0.5, AdamW 2e-4, cosine to 1%, batch 32, bf16
- Source checkpoint: experiments/out/exp16_cloud/run_20261013_0044/ckpt/final.pth
- Exam (5 unseen series, 150 pages): detail better than JaNai on 140/150; ~22x faster than the teacher on the laptop
- Known issues: smooths film grain / paper texture (L1 loss); slight colour wobble on blank areas
