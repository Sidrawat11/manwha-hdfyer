"""Exp 18: gutter skip. How much of each page is a blank full-width gap (gutter), and how many 512-row strips
would skipping gutters save? Measured only; gutter skip was not built into the pipeline (see README.md).

Self-contained: the detector, the segment cutting and the strip count are all in this file (the pipeline's
strip planner: 512-row strips, 32-row overlap, last strip anchored to the bottom, short pieces padded up).

Rule (Exp 18): a gutter is >= 64 consecutive full-width rows whose pixels all lie within +-4 levels per channel of
the run's median colour. Gradients and texture don't qualify. Each content segment keeps up to 16 gutter rows on
either side as context. For comparison, the Exp 4 rule: runs of >= 64 rows that are each flat on their own (spread
<= 8), even if the colour drifts from row to row.

Sets: bench_chapters_par/ (labelled "library series 1-3" in folder order) and the five exam sets in exams/
(labelled A-E from exams/series_key.json, which stays local). Series names never appear in the output.

  python experiments/exp18/measure_gutters.py          -> experiments/exp18/results.csv
"""
import csv, glob, json, os, time, zipfile
import cv2, numpy as np

STRIP, OVERLAP = 512, 32
MIN_ROWS, TOLERANCE, CONTEXT = 64, 4, 16
OUT = "experiments/exp18/results.csv"
IMG = (".jpg", ".jpeg", ".png", ".webp")


def pages(cbz):
    with zipfile.ZipFile(cbz) as z:
        for name in sorted(n for n in z.namelist() if n.lower().endswith(IMG) and not n.startswith(("__MACOSX", "."))):
            img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                yield img


def find_gutters(image, min_rows=MIN_ROWS, tolerance=TOLERANCE):
    """(start, end) of runs of >= min_rows full-width rows within +-tolerance of the run's median colour."""
    row_min = image.min(axis=1).astype(np.int16)
    row_max = image.max(axis=1).astype(np.int16)
    band = 2 * tolerance
    candidate = (row_max - row_min).max(axis=1) <= band
    gutters, h, y = [], image.shape[0], 0
    while y < h:
        if not candidate[y]:
            y += 1
            continue
        low, high, end = row_min[y].copy(), row_max[y].copy(), y + 1
        while end < h and candidate[end]:
            new_low, new_high = np.minimum(low, row_min[end]), np.maximum(high, row_max[end])
            if (new_high - new_low).max() > band:
                break
            low, high, end = new_low, new_high, end + 1
        if end - y >= min_rows:
            colour = np.round(np.median(image[y:end].reshape(-1, image.shape[2]), axis=0)).astype(np.int16)
            if max((high - colour).max(), (colour - low).max()) <= tolerance:
                gutters.append((y, end))
        y = end
    return gutters


def content_segments(height, gutters, context=CONTEXT):
    """(start, end) of each content segment between gutters, with up to `context` gutter rows on either side."""
    segments, y, above_rows = [], 0, 0
    for start, end in gutters + [(height, height)]:
        if start > y:
            segments.append((y - min(context, above_rows), start + min(context, end - start)))
        y, above_rows = end, end - start
    return segments


def strips_for(height, strip=STRIP, overlap=OVERLAP):
    """How many fixed-shape strips the pipeline cuts from `height` rows (a piece shorter than a strip is padded to one)."""
    count, y = 0, 0
    while y + strip < height:
        count, y = count + 1, y + strip - overlap
    return count + 1


def exp4_rows(image, tol=8, min_len=MIN_ROWS):
    spread = image.max(axis=1).astype(np.int16) - image.min(axis=1).astype(np.int16)
    flat = spread.max(axis=1) <= tol
    d = np.diff(np.concatenate([[0], flat.astype(np.int8), [0]]))
    lengths = np.where(d == -1)[0] - np.where(d == 1)[0]
    return int(lengths[lengths >= min_len].sum())


def main():
    sets = []
    library = sorted({os.path.basename(os.path.dirname(p)) for p in glob.glob("bench_chapters_par/*/*.cbz")})
    for i, series in enumerate(library):
        sets.append((f"library series {i + 1}", "", sorted(glob.glob(f"bench_chapters_par/{series}/*.cbz"))))
    key = json.load(open("exams/series_key.json"))
    by_series = {entry["series"]: (label, entry["style"]) for label, entry in key.items()}
    for d in sorted(glob.glob("exams/*/")):
        label, style = by_series[json.load(open(os.path.join(d, "manifest.json")))["series"]]
        sets.append((f"exam {label}", style, [os.path.join(d, "test.cbz")]))
    sets.sort(key=lambda s: s[0])

    rows = []
    for label, style, cbzs in sets:
        n = with_gutters = total_rows = gutter_rows = exp4 = today = skip = 0
        detect_s = 0.0
        for cbz in cbzs:
            for img in pages(cbz):
                h = img.shape[0]
                t = time.perf_counter(); gutters = find_gutters(img); detect_s += time.perf_counter() - t
                n += 1; total_rows += h; with_gutters += bool(gutters)
                gutter_rows += sum(e - s for s, e in gutters)
                exp4 += exp4_rows(img)
                segments = content_segments(h, gutters) if gutters else []
                today += strips_for(h)
                skip += sum(strips_for(e - s) for s, e in segments) if segments else strips_for(h)
        rows.append({"set": label, "style": style, "pages": n, "pages_with_gutters": with_gutters,
                     "gutter_rows_pct": round(100 * gutter_rows / total_rows, 1),
                     "exp4_rule_rows_pct": round(100 * exp4 / total_rows, 1),
                     "strips_today": today, "strips_with_gutter_skip": skip,
                     "strips_saved_pct": round(100 * (today - skip) / today, 1),
                     "detect_ms_per_page": round(1000 * detect_s / n, 1)})
        print(rows[-1], flush=True)
    total = {"set": "all", "style": "", **{k: sum(r[k] for r in rows) for k in ("pages", "pages_with_gutters", "strips_today", "strips_with_gutter_skip")}}
    total["strips_saved_pct"] = round(100 * (total["strips_today"] - total["strips_with_gutter_skip"]) / total["strips_today"], 1)
    total["detect_ms_per_page"] = round(sum(r["detect_ms_per_page"] * r["pages"] for r in rows) / total["pages"], 1)
    rows.append(total)
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
