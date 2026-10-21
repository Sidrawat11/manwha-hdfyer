# Exp 21: sharpening bake-off on full pages

**Question.** Can a sharpening filter make the student (hdfyer-2x-s-r2, no finishing) look as good as the teacher (4x-UltraSharp halved to 2x)?

**Verdict.** On edges, yes; on texture, no.
- **Edges.** The raw student has 88% of UltraSharp's edge acutance. Edge-masked unsharp masking (amount 1.0, radius 0.8) brings it to 103% without adding any noise in flat areas. Today's production finish reaches 93%.
- **Texture.** No filter brings the student closer to UltraSharp pixel by pixel. Every variant sits at or above the raw student's distance (2.76 levels), because what separates the two is texture UltraSharp invents: fibre, grain, the D overlay. Sharpening can only amplify what the student drew.
- **What that means.** On line art and flat colour, a sharpened student is hard to tell from UltraSharp. On textured art, closing the gap needs the student itself to draw more detail; a better filter won't do it.

## Setup

- **Pages.** 20 full pages listed in `pages.json`: three each from training L1–L3 and exam D, two each from exams A, B, C and E. They cover line art, faces, lettering and SFX, gradients, texture overlays and flat colour. They were chosen from content measures and then checked by eye; scanlation credit pages were excluded.
- **Sources.**
  - Student: `data/student/hdfyer-2x-s-r2-raw/`, made by `make_student.py`. It is production's digest stage exactly (TensorRT FP16, 512-row strips, linear-blend stitch), with no sharpen and no cleanup.
  - Teacher: `data/teacher/ultrasharp-2x/`.
- **Variants.** 17 in `sharpen.py`, applied to both sources:

| Variant | What it does | Settings |
|---|---|---|
| plain | nothing | – |
| baseline | today's production finish: USM, then near-black cleanup | strength 0.3, radius 0.5 |
| usm_edge | luma USM, applied only on edges | (0.6, r0.8), (1.0, r0.8), (1.2, r1.2) |
| cas | AMD FidelityFX CAS | sharpness 0.3, 0.6, 1.0 |
| warp | aWarpSharp-style warp | depth 2, 4, 8 |
| rl | Richardson–Lucy deconvolution, gaussian PSF | (σ0.7, 10 it), (σ1.0, 10 it), (σ1.0, 25 it) |
| guided | guided-filter detail boost | (r2, ×1), (r2, ×2), (r8, ε 0.01, ×1) |

- **Scores.** `run.py` scores on luma. The places it measures (edges, a 3 px band around them, flat areas) all come from plain UltraSharp, so both sources are measured at the same spots. Overshoot is measured against each variant's own unsharpened source. Full definitions are in the `run.py` docstring.
  - Per page × source × variant: `scores.csv`
  - Means over the pages: `summary.csv`

## Results (student; means over 20 pages)

| Variant | ms/Mpx | Acutance vs UltraSharp | Overshoot (levels) | Band >6 levels | Noise gain | Distance to UltraSharp |
|---|---|---|---|---|---|---|
| plain | 0 | 0.884 | 0.00 | 0.0% | 1.00 | 2.76 |
| baseline (production) | 15 | 0.926 | 0.39 | 1.4% | 1.06 | 2.76 |
| **usm_edge 1.0 r0.8** | 12 | **1.032** | 0.83 | 4.4% | **1.00** | 2.93 |
| usm_edge 1.2 r1.2 | 14 | 1.122 | 1.94 | 9.2% | 1.00 | 3.19 |
| cas 0.6 | 9 | 1.008 | 0.58 | 3.2% | 1.41 | 2.87 |
| warp d4 | 18 | 0.867 | 0.00 | 0.0% | 1.00 | 2.85 |
| rl σ1.0 10 it | 26 | 1.270 | 4.38 | 20.0% | 1.18 | 4.20 |
| guided r2 ×1 | 6 | 0.923 | 0.19 | 0.2% | 1.74 | 2.80 |

**Plain UltraSharp scores 1.000 acutance and 0 distance** by construction. Sharpening UltraSharp itself with the production finish takes it to 1.047.

### Acutance by set (student, vs plain UltraSharp)

| Variant | L1 | L2 | L3 | A | B | C | D | E |
|---|---|---|---|---|---|---|---|---|
| plain | 0.90 | 0.88 | 0.92 | 0.89 | 0.90 | 0.86 | 0.84 | 0.90 |
| baseline | 0.94 | 0.92 | 0.97 | 0.93 | 0.94 | 0.91 | 0.88 | 0.94 |
| usm_edge 1.0 r0.8 | 1.05 | 1.02 | 1.06 | 1.09 | 1.04 | 1.00 | 0.96 | 1.06 |
| cas 0.6 | 1.04 | 0.99 | 1.03 | 1.00 | 0.97 | 0.98 | 1.04 | 0.99 |

D (the gritty texture overlay) is the hardest set for the student. There, CAS gets closer than edge-masked USM because CAS also lifts the student's faint grain.

## Notes

- **Edge-masked USM** is the only method that raises acutance to UltraSharp's level without adding noise in flat areas. It leaves flat colour and gradients alone, and it puts the same luma detail on all channels, so it adds no colour fringes. Its cost is overshoot at edges: 0.83 levels on average, with 4.4% of the edge band past 6 levels. Look for halos around lettering in `compare.html` before choosing it.
- **CAS** is the fastest real sharpener, at about 9 ms/Mpx. It sharpens everything, though, so flat-area noise goes up 1.4×.
- **Richardson–Lucy** gives the most acutance but the worst halos: 20% of the edge band over 6 levels at σ1.0.
- **Guided detail boost** amplifies texture and noise, not edges. That is how He et al.'s detail enhancement works.
- **Warp** never adds overshoot or noise, but it scores *lower* acutance on these pages. Thinning lines that are only 1–3 px wide also lowers their contrast. Its distance to UltraSharp grows with depth.
- **The baseline's noise gain of 1.06** combines a small rise from the USM with the near-black cleanup flattening dark areas.
- **Timings** cover CPU→GPU→CPU on an RTX 4060 Laptop GPU, medians over the 20 pages. The baseline is OpenCV on the CPU. A typical 2x page is about 14 Mpx, so edge-masked USM costs about 170 ms per page, against about 210 ms for today's finish (which production runs on 4 CPU threads, overlapped with the GPU).

## Look at it

`outputs/exp21-sharpening/compare.html` (local, gitignored, about 6.6 GB of images):
- Every page at actual size, with a slider between any two of the 34 variants.
- A 2× zoom.
- Winner buttons; your picks download as JSON.

It opens on student baseline against plain UltraSharp.

## Rerun

```bash
python experiments/exp21/make_student.py   # once: the student reference set, about 22 min for 728 pages
python experiments/exp21/run.py            # about 50 min: images, scores.csv, summary.csv, compare.html
```
