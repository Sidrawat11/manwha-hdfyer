"""Rebuild every dataset in data/ from the scanlation originals in data/library/, and check it.

  build    rebuilds data/training/v3/, data/exams/A..E/ and data/bench/ from data/library/, using
             data/training/v3/recipe.json   the training set's recipe (seed, arguments, series, reserved chapters,
                                            held-out test pages, and the expected page list)
             data/exams/manifest.json       the exam pages (source chapter and page for every entry)
             data/bench/manifest.json       which library chapter each bench chapter is (path + sha256)
  verify   checks the rebuilt sets against those files: the training page list against the recipe's expected list,
           the exam CBZs against their manifest, the bench chapters against their sha256

The recipe and manifests were recorded on 2026-10-18 from the folders data/ replaced; at the time the rebuilt sets were
verified page by page and byte for byte against them. Training series are labelled L1-L3 and exam series A-E; the
names live only in data/ (gitignored).

Run from the repo root: python experiments/data/rebuild_datasets.py build|verify
"""
import argparse, glob, hashlib, json, os, random, re, shutil, sys, zipfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp16"))
from tqdm import tqdm

LIBRARY = "data/library"
TRAIN = "data/training/v3"
EXAMS = "data/exams"
BENCH = "data/bench"


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def rel(cbz_path):
    """Any path to a library chapter -> '<series>/<chapter>.cbz' (location independent)."""
    parts = cbz_path.replace("\\", "/").split("/")
    return "/".join(parts[-2:])


def write_cbz(path, entries):
    """entries: [(entry name, bytes)] -> a stored (uncompressed) zip, the way make_dataset.py writes them."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name, data in entries:
            z.writestr(name, data)


def read_page(cbz_rel, page):
    with zipfile.ZipFile(os.path.join(LIBRARY, cbz_rel)) as z:
        return z.read(page)


# ---------------------------------------------------------------- build

def build():
    import make_dataset as md
    recipe = json.load(open(f"{TRAIN}/recipe.json"))
    reserved = set(recipe["reserved_chapters"])
    rng = random.Random(recipe["seed"])
    manifest = {"seed": recipe["seed"], "recipe": "recipe.json", "train": {}, "test": []}
    for series in tqdm(recipe["series"], desc="training series"):
        label = recipe["labels"][series]
        cbzs = [c for c in sorted(glob.glob(os.path.join(LIBRARY, series, "*.cbz"))) if rel(c) not in reserved]
        rng.shuffle(cbzs)
        train, train_cbzs = [], []
        for cbz in cbzs:
            if len(train) >= recipe["train_per_series"]:
                break
            got = md.pick_page(cbz, rng)
            if got is None:
                continue
            train.append((rel(cbz), got[0], got[1])); train_cbzs.append(cbz)
        for cbz in train_cbzs[:recipe["extra_per_series"]]:   # a second, different page from some chapters
            names = [n for n in md.pages_of(cbz) if n not in {p for c, p, _ in train if c == rel(cbz)}]
            if names:
                n = rng.choice(names)
                with zipfile.ZipFile(cbz) as z:
                    train.append((rel(cbz), n, z.read(n)))
        entries = [(f"{k:03d}{os.path.splitext(p)[1].lower()}", data) for k, (_, p, data) in enumerate(train)]
        write_cbz(f"{TRAIN}/train_{label}.cbz", entries)
        manifest["train"][f"train_{label}.cbz"] = [{"entry": e, "series_label": label, "cbz": c, "page": p}
                                                   for (e, _), (c, p, _) in zip(entries, train)]
    test = [(e["entry"], read_page(e["cbz"], e["page"])) for e in recipe["test"]]
    write_cbz(f"{TRAIN}/test.cbz", test)
    manifest["test"] = [{"entry": e["entry"], "series_label": recipe["labels"][e["series"]], "cbz": e["cbz"], "page": e["page"]} for e in recipe["test"]]
    json.dump(manifest, open(f"{TRAIN}/manifest.json", "w"), indent=1)

    exams = json.load(open(f"{EXAMS}/manifest.json"))
    for label, s in tqdm(exams["sets"].items(), desc="exam sets"):
        os.makedirs(f"{EXAMS}/{label}", exist_ok=True)
        write_cbz(f"{EXAMS}/{label}/test.cbz", [(e["entry"], read_page(e["cbz"], e["page"])) for e in s["test"]])
        json.dump({"label": label, "seed": s["seed"], "series": s["series"], "test": s["test"]}, open(f"{EXAMS}/{label}/manifest.json", "w"), indent=1)

    bench = json.load(open(f"{BENCH}/manifest.json"))
    for set_name, items in bench["sets"].items():
        for it in tqdm(items, desc=f"bench {set_name}"):
            dst = os.path.join(BENCH, set_name, it["label"], os.path.basename(it["cbz"]))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(LIBRARY, it["cbz"]), dst)
    print("built", TRAIN, EXAMS, BENCH)


# ---------------------------------------------------------------- verify

def entries(path):
    with zipfile.ZipFile(path) as z:
        return [(n, z.read(n)) for n in z.namelist()]


def verify():
    problems = []
    recipe = json.load(open(f"{TRAIN}/recipe.json")); v3 = json.load(open(f"{TRAIN}/manifest.json"))
    for label, expected in recipe["expected"]["train"].items():
        listed = [[e["cbz"], e["page"]] for e in v3["train"][f"train_{label}.cbz"]]
        in_cbz = len(entries(f"{TRAIN}/train_{label}.cbz"))
        ok = listed == expected and in_cbz == len(expected)
        print(f"training {label}: {in_cbz} pages | page list as recorded: {listed == expected}")
        if not ok: problems.append(f"training {label}")
    test = entries(f"{TRAIN}/test.cbz")
    ok = [n for n, _ in test] == [e["entry"] for e in recipe["test"]] and all(
        data == read_page(e["cbz"], e["page"]) for (_, data), e in zip(test, recipe["test"]))
    print(f"training test: {len(test)} pages | entries match their source pages: {ok}")
    if not ok: problems.append("training test")

    exams = json.load(open(f"{EXAMS}/manifest.json"))
    for label, s in exams["sets"].items():
        got = entries(f"{EXAMS}/{label}/test.cbz")
        ok = [n for n, _ in got] == [e["entry"] for e in s["test"]] and all(
            data == read_page(e["cbz"], e["page"]) for (_, data), e in zip(got, s["test"]))
        print(f"exam {label}: {len(got)} pages | entries match their source pages: {ok}")
        if not ok: problems.append(f"exam {label}")

    bench = json.load(open(f"{BENCH}/manifest.json"))
    for set_name, items in bench["sets"].items():
        ok = sum(sha256(os.path.join(BENCH, set_name, it["label"], os.path.basename(it["cbz"]))) == it["sha256"] for it in items)
        print(f"bench {set_name}: {ok}/{len(items)} chapters match their recorded sha256")
        if ok != len(items): problems.append(f"bench {set_name}")
    print("ALL AS RECORDED" if not problems else "DIFFERENCES: " + ", ".join(problems))
    return not problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("step", choices=["build", "verify"])
    step = ap.parse_args().step
    if step == "build": build()
    else: sys.exit(0 if verify() else 1)
