"""Exp 16 dataset: random training and test pages, packed into small CBZs you can upload.

  training/train_<series>.cbz   60 pages per series, one random page from each of 60 random chapters
  training/test.cbz             10 pages per series from chapters NOT used for training (held out)
  training/manifest.json        where every page came from (series, chapter, page), seed, and the split
Chapters used in Exp 14/15 are kept out of both, so those 10 pages stay a fair third check
(the student can later be compared with the Exp 15 models on exactly the same pages).
Pages under 300 px tall (credits, separators) are skipped. JPEG bytes are copied as-is: no re-encoding.

Run from the repo root:  python experiments/exp16/make_dataset.py
Bigger set that keeps the SAME test pages (so runs stay comparable):
  python experiments/exp16/make_dataset.py --out training_v2 --train-per-series 999 --extra-per-series 50 --reuse-test training
  (999 = one page from every available chapter; extra = a second page from that many chapters)

Exam set for a series the student never saw (one random page per chapter, nothing else touched):
  python experiments/exp16/make_dataset.py --exam "<series folder>"
  -> exams/<series slug>/test.cbz + manifest.json  (then: distill.py exam --data exams/<series slug> ...)
"""
import argparse, shutil
import glob, io, json, os, random, re, sys, zipfile
import cv2, numpy as np

SEED, TRAIN_PER_SERIES, TEST_PER_SERIES = 2030, 60, 10
OUT = "training"
IMG = re.compile(r"\.(jpe?g|png|webp)$", re.I)


def pages_of(cbz):
    with zipfile.ZipFile(cbz) as z:
        return sorted(n for n in z.namelist() if IMG.search(n) and not n.endswith("/"))


def pick_page(cbz, rng):
    """A random page at least 300 px tall; returns (name, bytes) or None."""
    names = pages_of(cbz)
    rng.shuffle(names)
    with zipfile.ZipFile(cbz) as z:
        for n in names[:8]:
            b = z.read(n)
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
            if img is not None and img.shape[0] * 2 >= 300:
                return n, b
    return None


def make_exam(series, pages, seed):
    """One random page (>= 300 px tall) from each of up to `pages` chapters of one series."""
    cbzs = sorted(glob.glob(os.path.join("Manwhas", series, "*.cbz")))
    if not cbzs:
        sys.exit(f"no CBZs in Manwhas/{series}")
    used = set()  # every chapter any training set has touched: the exam must be unseen
    for m in glob.glob("training*/manifest.json"):
        j = json.load(open(m))
        used |= {e["cbz"] for v in j["train"].values() for e in v} | {e["cbz"] for e in j["test"]}
    cbzs = [c for c in cbzs if c not in used]
    rng = random.Random(seed)
    rng.shuffle(cbzs)
    slug = re.sub(r"[^a-z0-9]+", "_", series.lower()).strip("_")
    out_dir = os.path.join("exams", slug); os.makedirs(out_dir, exist_ok=True)
    items = []
    for cbz in cbzs:
        if len(items) >= pages:
            break
        got = pick_page(cbz, rng)
        if got:
            items.append({"series": series, "cbz": cbz, "page": got[0], "bytes": got[1]})
    items.sort(key=lambda it: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", it["cbz"])])  # chapter order
    manifest = {"seed": seed, "series": series, "test": []}
    with zipfile.ZipFile(os.path.join(out_dir, "test.cbz"), "w", zipfile.ZIP_STORED) as z:
        for k, it in enumerate(items):
            ext = os.path.splitext(it["page"])[1].lower()
            z.writestr(f"{k:03d}{ext}", it["bytes"])
            manifest["test"].append({"entry": f"{k:03d}{ext}", "series": series, "cbz": it["cbz"], "page": it["page"]})
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    mb = os.path.getsize(os.path.join(out_dir, "test.cbz")) / 1e6
    print(f"exam {series}: {len(items)} pages from {len(items)} chapters ({len(used)} training chapters excluded) "
          f"-> {out_dir}/test.cbz ({mb:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--train-per-series", type=int, default=TRAIN_PER_SERIES)
    ap.add_argument("--extra-per-series", type=int, default=0, help="second page from this many training chapters")
    ap.add_argument("--reuse-test", default="", help="folder whose test.cbz + manifest test entries are reused")
    ap.add_argument("--exam", default="", help="series folder name: build an unseen exam set instead")
    ap.add_argument("--exam-pages", type=int, default=30)
    a = ap.parse_args()
    if a.exam:
        return make_exam(a.exam, a.exam_pages, SEED)
    out_dir = a.out
    reserved = set()
    for f in ("experiments/out/exp14_tensorrt/pages.json",):
        if os.path.exists(f):
            reserved |= {p["cbz"] for p in json.load(open(f))}
    reuse = None
    if a.reuse_test:
        reuse = json.load(open(os.path.join(a.reuse_test, "manifest.json")))["test"]
        reserved |= {e["cbz"] for e in reuse}
    rng = random.Random(SEED)
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"seed": SEED, "reserved_exp14_chapters": sorted(reserved), "train": {}, "test": []}
    test_items = []
    for series in sorted(os.listdir("Manwhas")):
        cbzs = [c for c in sorted(glob.glob(os.path.join("Manwhas", series, "*.cbz"))) if c not in reserved]
        rng.shuffle(cbzs)
        slug = re.sub(r"[^a-z0-9]+", "_", series.lower()).strip("_")
        train, test, used = [], [], 0
        n_test = 0 if reuse else TEST_PER_SERIES
        train_cbzs = []
        for cbz in cbzs:
            if len(train) >= a.train_per_series and len(test) >= n_test:
                break
            got = pick_page(cbz, rng)
            if got is None:
                continue
            item = {"series": series, "cbz": cbz, "page": got[0], "bytes": got[1]}
            if len(train) < a.train_per_series:
                train.append(item); train_cbzs.append(cbz)
            else:
                test.append(item)
        for cbz in train_cbzs[:a.extra_per_series]:  # a second, different page from some chapters
            names = [n for n in pages_of(cbz) if n not in {t["page"] for t in train if t["cbz"] == cbz}]
            if names:
                n = rng.choice(names)
                with zipfile.ZipFile(cbz) as z:
                    train.append({"series": series, "cbz": cbz, "page": n, "bytes": z.read(n)})
        path = os.path.join(out_dir, f"train_{slug}.cbz")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
            for k, it in enumerate(train):
                ext = os.path.splitext(it["page"])[1].lower()
                z.writestr(f"{k:03d}{ext}", it["bytes"])
        manifest["train"][path] = [{"entry": f"{k:03d}{os.path.splitext(it['page'])[1].lower()}",
                                    "series": it["series"], "cbz": it["cbz"], "page": it["page"]} for k, it in enumerate(train)]
        test_items += test
        print(f"{series}: {len(train)} train, {len(test)} test, {os.path.getsize(path) / 1e6:.0f} MB", flush=True)
    path = os.path.join(out_dir, "test.cbz")
    if reuse:
        shutil.copyfile(os.path.join(a.reuse_test, "test.cbz"), path)
        manifest["test"] = reuse; test_items = reuse
    else:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
            for k, it in enumerate(test_items):
                ext = os.path.splitext(it["page"])[1].lower()
                z.writestr(f"{k:03d}{ext}", it["bytes"])
                manifest["test"].append({"entry": f"{k:03d}{ext}", "series": it["series"], "cbz": it["cbz"], "page": it["page"]})
    # sanity: no chapter in both
    train_ch = {e["cbz"] for v in manifest["train"].values() for e in v}
    test_ch = {e["cbz"] for e in manifest["test"]}
    exp14 = {c for c in reserved if c not in test_ch}
    assert not (train_ch & test_ch) and not (train_ch & reserved) and not (test_ch & exp14), "chapter overlap!"
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    total = sum(os.path.getsize(f) for f in glob.glob(os.path.join(out_dir, "*.cbz")))
    print(f"test: {len(test_items)} pages | total upload {total / 1e6:.0f} MB | no chapter overlap: OK")


if __name__ == "__main__":
    main()
