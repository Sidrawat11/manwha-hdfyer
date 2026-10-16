# Exp 18: gutter skip (measured, not shipped)

**Question.** Gutters are the blank full-width gaps between panels. On the laptop the GPU is now the bottleneck, so not sending gutter rows to the model would be a direct speed-up. How many strips would that actually save?

**Verdict.** Not shipped. The ceiling is about 3% fewer strips: 2.8% overall, 5.3% at best. Detecting gutters costs about 85 ms per page of CPU time on the pipeline's main thread, which is more than the GPU time it would save.

## The rule

A gutter is at least **64 consecutive full-width rows** whose pixels all lie within **±4 levels per channel of the run's median colour**. The tolerance absorbs JPEG noise; gradients and texture don't qualify. The plan was to cut the page into the content segments between gutters, keep up to 16 gutter rows on either side of each segment as context, and cut each segment with the usual 512-row strips. Stitching would then paint each gutter back in at its colour.

## Results

Pages from `bench_chapters_par/` (three library series, 16 chapters) and the five exam sets (30 pages each). The script is `measure_gutters.py` and the numbers are in `results.csv`.

| Set | Pages | Pages with gutters | Gutter rows | Strips today | With gutter skip | Saved |
|---|---|---|---|---|---|---|
| library series 1 | 210 | 63 | 2.6% | 2,001 | 1,979 | 1.1% |
| library series 2 | 273 | 68 | 2.9% | 2,329 | 2,276 | 2.3% |
| library series 3 | 310 | 155 | 10.4% | 1,705 | 1,614 | **5.3%** |
| exam A (soft pastel, painted) | 30 | 17 | 8.2% | 164 | 156 | 4.9% |
| exam B (clean modern line art) | 30 | 13 | 8.9% | 158 | 151 | 4.4% |
| exam C (sharp action line art) | 30 | 23 | 9.5% | 222 | 214 | 3.6% |
| exam D (gritty texture overlay) | 30 | 0 | 0.0% | 84 | 84 | 0.0% |
| exam E (manhua, heavy gradients) | 30 | 12 | 2.9% | 120 | 118 | 1.7% |
| **All** | **943** | **351** | | **6,783** | **6,592** | **2.8%** |

Fewer strips are saved than there are gutter rows. Each segment carries its context rows, and a short segment is padded up to a full 512-row strip.

**Detection cost:** about 85 ms per page (22–99 ms depending on the set). Almost all of it is the per-row minimum and maximum over the whole page. On `bench_chapters_par/` that comes to roughly 66 s of main-thread CPU time, against a few seconds of GPU time saved. Faster detection is possible, but even at zero cost the gain stays below about 3%.

## Why Exp 4 saw about 15%

Exp 4 counted runs of rows that are each flat on their own (spread of 8 levels or less), even when the colour drifts from row to row. Most of those rows are gradients and fades, which painting one flat colour would visibly change, so this rule excludes them.

| Set | Exp 4 rule (each row flat, colour may drift) | Exp 18 rule (one colour, ±4) |
|---|---|---|
| library series 1 | 17.3% | 2.6% |
| library series 2 | 30.9% | 2.9% |
| library series 3 | 23.9% | 10.4% |
| exam A | 18.1% | 8.2% |
| exam B | 31.2% | 8.9% |
| exam C | 19.7% | 9.5% |
| exam D | 0.0% | 0.0% |
| exam E | 8.0% | 2.9% |

## Rerun

```bash
python experiments/exp18/measure_gutters.py
```

It needs `bench_chapters_par/` and `exams/` with its local `series_key.json`; all of these are gitignored. It writes `results.csv`.
