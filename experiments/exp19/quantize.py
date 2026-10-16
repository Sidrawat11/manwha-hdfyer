"""Exp 19 step 2: INT8 (explicit Q/DQ) versions of hdfyer, made with NVIDIA ModelOpt ONNX post-training quantization.

TensorRT 11 has no implicit INT8 calibration (no BuilderFlag.INT8, no IInt8EntropyCalibrator2), so INT8 comes from
Q/DQ nodes in the ONNX file; engines are then built strongly typed, like production's FP16 ones.

  1. FP32 ONNX export of hdfyer (the folded weights, same values as production's FP16 export), with the width free so
     one quantized model serves every page width (height fixed at the production strip height, 512).
  2. Calibration set: 300 strips of 512x800 from the library series in bench_chapters_par/ (not the exam series, which
     stay unseen; Chapter 134, the speed-test chapter, is left out too). Strips from wider pages are cropped to 800.
  3. Two variants:  int8-default  ModelOpt's default Q/DQ placement (entropy calibration)
                    int8-mixed    the same, but the first conv, the last conv and the pixel-shuffle stay FP16

ModelOpt lives in experiments/out/exp19/pylib (pip --target, see README): its [onnx] extra pins onnx 1.21, while the
pipeline uses 1.23.2, so it is kept out of the main environment. Run with:
  PYTHONPATH=experiments/out/exp19/pylib python experiments/exp19/quantize.py
"""
import copy, glob, os, random, sys, zipfile
sys.path.insert(0, os.getcwd())
import cv2, numpy as np, torch
from tqdm import tqdm
from hdfyer.digest.model import load_student

WEIGHTS = "models/hdfyer/hdfyer-2x-s-r2.pth"
OUT = "experiments/out/exp19"
STRIP, CAL_WIDTH, CAL_STRIPS, SEED = 512, 800, 300, 2031
EXCLUDE_CHAPTER = "Chapter 134_"
MIXED_FP16_NODES = ["/conv_1/eval_conv/Conv", "/upsampler/upsampler.0/Conv", "/upsampler/upsampler.1/DepthToSpace"]


def export_fp32(path):
    model = copy.deepcopy(load_student(WEIGHTS)).float().eval()
    with torch.no_grad():
        torch.onnx.export(model, torch.zeros((1, 3, STRIP, CAL_WIDTH), device="cuda"), path, opset_version=17,
                          input_names=["x"], output_names=["y"], dynamo=False,
                          dynamic_axes={"x": {3: "width"}, "y": {3: "width2"}})


def calibration_strips():
    """CAL_STRIPS random 512-row strips (RGB, float32 0-1, NCHW) cut from the library chapters."""
    rng = random.Random(SEED)
    cbzs = [c for c in sorted(glob.glob("bench_chapters_par/*/*.cbz")) if EXCLUDE_CHAPTER not in c]
    candidates = []
    for cbz in cbzs:
        with zipfile.ZipFile(cbz) as z:
            candidates += [(cbz, n) for n in sorted(z.namelist()) if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    rng.shuffle(candidates)
    strips = []
    with tqdm(total=CAL_STRIPS, desc="calibration strips") as bar:
        for cbz, name in candidates:
            with zipfile.ZipFile(cbz) as z:
                img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.shape[0] < STRIP or img.shape[1] < CAL_WIDTH:
                continue
            y = rng.randrange(0, img.shape[0] - STRIP + 1); x = rng.randrange(0, img.shape[1] - CAL_WIDTH + 1)
            strip = cv2.cvtColor(img[y:y + STRIP, x:x + CAL_WIDTH], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            strips.append(strip.transpose(2, 0, 1)); bar.update(1)
            if len(strips) == CAL_STRIPS:
                break
    return np.stack(strips), len(cbzs)


def main():
    import modelopt.onnx.quantization as moq
    os.makedirs(OUT, exist_ok=True)
    fp32 = f"{OUT}/hdfyer_fp32_dynamic_width.onnx"
    if not os.path.exists(fp32):
        export_fp32(fp32)
    data, chapters = calibration_strips()
    print(f"calibration: {data.shape[0]} strips of {STRIP}x{CAL_WIDTH} from {chapters} library chapters", flush=True)
    for variant, exclude in [("int8-default", None), ("int8-mixed", MIXED_FP16_NODES)]:
        out = f"{OUT}/hdfyer_{variant}.onnx"
        if os.path.exists(out):
            continue
        print(f"quantizing {variant} ...", flush=True)
        moq.quantize(fp32, quantize_mode="int8", calibration_data={"x": data}, calibration_method="entropy",
                     calibration_eps=["cuda:0", "cpu"], calibration_shapes=f"x:1x3x{STRIP}x{CAL_WIDTH}",
                     high_precision_dtype="fp16", nodes_to_exclude=exclude, output_path=out, log_level="WARNING")
    print("done")


if __name__ == "__main__":
    main()
