"""Exp 21 sharpeners. Each takes a uint8 BGR page (numpy) and returns one; all but the production baseline run in
torch on the GPU (float32, 0-255). VARIANTS lists every setting the bake-off runs.

    baseline   the production finish when Exp 21 ran: unsharp mask (strength 0.3, radius 0.5) + near-black cleanup
    usm_edge   unsharp mask on luma, applied only where there are edges (flat areas and gradients left alone)
    cas        AMD FidelityFX Contrast Adaptive Sharpening (the no-scaling CAS filter, per channel)
    warp       aWarpSharp-style: pulls pixels away from edge ridges along the gradient of a blurred edge map, so
               transitions get narrower without adding overshoot
    rl         Richardson-Lucy deconvolution with a gaussian PSF (per channel)
    guided     guided-filter detail boost (He et al.): self-guided edge-preserving base, the residual amplified. The
               residual is what varies less than ~sqrt(eps) inside the window, so texture and noise rise with it
"""
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LUMA = (0.114, 0.587, 0.299)  # BGR


def to_t(img):
    return torch.from_numpy(np.ascontiguousarray(img)).to(DEV).permute(2, 0, 1)[None].float()


def from_t(x):
    return x.clamp(0, 255).round().byte()[0].permute(1, 2, 0).contiguous().cpu().numpy()


def luma(x):
    return (x[:, 0:1] * LUMA[0] + x[:, 1:2] * LUMA[1] + x[:, 2:3] * LUMA[2])


def gauss_kernel(sigma):
    r = max(1, int(np.ceil(3 * sigma)))
    k = torch.exp(-torch.arange(-r, r + 1, device=DEV, dtype=torch.float32) ** 2 / (2 * sigma * sigma))
    return k / k.sum()


def blur(x, sigma):
    """Separable gaussian, reflect padding, any number of channels."""
    k = gauss_kernel(sigma)
    r, c = len(k) // 2, x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def box(x, r):
    return F.avg_pool2d(x, 2 * r + 1, stride=1, padding=r, count_include_pad=False)


def sobel(y):
    yp = F.pad(y, (1, 1, 1, 1), mode="replicate")
    gx = (yp[..., 1:-1, 2:] - yp[..., 1:-1, :-2]) * 2 + (yp[..., :-2, 2:] - yp[..., :-2, :-2]) + (yp[..., 2:, 2:] - yp[..., 2:, :-2])
    gy = (yp[..., 2:, 1:-1] - yp[..., :-2, 1:-1]) * 2 + (yp[..., 2:, :-2] - yp[..., :-2, :-2]) + (yp[..., 2:, 2:] - yp[..., :-2, 2:])
    return gx / 8, gy / 8  # levels per pixel


# ---------------------------------------------------------------- the sharpeners
def plain(img):
    return img


def baseline(img, strength=0.3, radius=0.5, threshold=15):
    """The production finish when Exp 21 ran (frozen here; production now uses usm_edge 1.0 r0.8): a plain unsharp
    mask on all three channels, then pixels with every channel below 15 set to black."""
    import cv2
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    out = np.clip(cv2.addWeighted(img, 1 + strength, blurred, -strength, 0), 0, 255).astype(np.uint8)
    out[np.all(out < threshold, axis=2)] = 0
    return out


@torch.no_grad()
def usm_edge(img, amount=1.0, radius=0.8, lo=6.0, hi=30.0):
    """Luma USM where the edge strength (gradient of the lightly blurred luma, levels/px) is between lo and hi
    (smoothstep), zero in flat areas; the same luma detail is added to all three channels (no colour fringes)."""
    x = to_t(img)
    y = luma(x)
    detail = y - blur(y, radius)
    gx, gy = sobel(blur(y, 0.7))
    t = ((torch.hypot(gx, gy) - lo) / (hi - lo)).clamp(0, 1)
    mask = blur(t * t * (3 - 2 * t), 1.0)
    return from_t(x + amount * mask * detail)


@torch.no_grad()
def cas(img, sharpness=0.5):
    """FidelityFX CAS (CasFilter without scaling), per channel on 0-1 values."""
    x = to_t(img) / 255
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    a, b, c = p[..., :-2, :-2], p[..., :-2, 1:-1], p[..., :-2, 2:]
    d, e, f = p[..., 1:-1, :-2], p[..., 1:-1, 1:-1], p[..., 1:-1, 2:]
    g, h, i = p[..., 2:, :-2], p[..., 2:, 1:-1], p[..., 2:, 2:]
    cross_mn = torch.minimum(torch.minimum(torch.minimum(d, e), torch.minimum(f, b)), h)
    cross_mx = torch.maximum(torch.maximum(torch.maximum(d, e), torch.maximum(f, b)), h)
    mn = cross_mn + torch.minimum(cross_mn, torch.minimum(torch.minimum(a, c), torch.minimum(g, i)))
    mx = cross_mx + torch.maximum(cross_mx, torch.maximum(torch.maximum(a, c), torch.maximum(g, i)))
    amp = (torch.minimum(mn, 2 - mx) / mx.clamp_min(1e-5)).clamp(0, 1).sqrt()
    w = amp * (-1 / (8 - 3 * sharpness))
    out = ((b + d + f + h) * w + e) / (1 + 4 * w)
    return from_t(out.clamp(0, 1) * 255)


@torch.no_grad()
def warp(img, depth=4.0, blur_sigma=2.0, thresh=64.0, max_px=2.0):
    """aWarpSharp-style: edge map = clamp(|sobel(luma)| / thresh), blurred; every pixel is resampled from
    p - depth * grad(edge map) (at most max_px away), i.e. from further out of each transition."""
    x = to_t(img)
    gx, gy = sobel(luma(x))
    em = blur((torch.hypot(gx, gy) / thresh).clamp(0, 1), blur_sigma)
    ex, ey = sobel(em)
    dx, dy = (-depth * ex).clamp(-max_px, max_px), (-depth * ey).clamp(-max_px, max_px)
    h, w = x.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=DEV, dtype=torch.float32), torch.arange(w, device=DEV, dtype=torch.float32), indexing="ij")
    grid = torch.stack(((xx + dx[0, 0]) / (w - 1) * 2 - 1, (yy + dy[0, 0]) / (h - 1) * 2 - 1), -1)[None]
    return from_t(F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True))


@torch.no_grad()
def rl(img, sigma=0.8, iterations=10):
    """Richardson-Lucy with a gaussian PSF (symmetric, so the flipped PSF is the same blur)."""
    d = to_t(img) + 1.0  # keep away from zero
    u = d.clone()
    for _ in range(iterations):
        u = u * blur(d / blur(u, sigma).clamp_min(1e-3), sigma)
    return from_t(u - 1.0)


@torch.no_grad()
def guided(img, radius=2, eps=1e-3, boost=1.0):
    """Self-guided filter per channel (on 0-1 values) as the base; out = I + boost * (I - base)."""
    x = to_t(img) / 255
    m = box(x, radius)
    var = box(x * x, radius) - m * m
    a = var / (var + eps)
    base = box(a, radius) * x + box(m - a * m, radius)
    return from_t((x + boost * (x - base)) * 255)


VARIANTS = {
    "plain": (plain, {}),
    "baseline": (baseline, {}),
    "usm_edge-0.6r0.8": (usm_edge, {"amount": 0.6, "radius": 0.8}),
    "usm_edge-1.0r0.8": (usm_edge, {"amount": 1.0, "radius": 0.8}),
    "usm_edge-1.2r1.2": (usm_edge, {"amount": 1.2, "radius": 1.2}),
    "cas-0.3": (cas, {"sharpness": 0.3}),
    "cas-0.6": (cas, {"sharpness": 0.6}),
    "cas-1.0": (cas, {"sharpness": 1.0}),
    "warp-d2": (warp, {"depth": 2.0}),
    "warp-d4": (warp, {"depth": 4.0}),
    "warp-d8": (warp, {"depth": 8.0}),
    "rl-s0.7i10": (rl, {"sigma": 0.7, "iterations": 10}),
    "rl-s1.0i10": (rl, {"sigma": 1.0, "iterations": 10}),
    "rl-s1.0i25": (rl, {"sigma": 1.0, "iterations": 25}),
    "guided-r2b1.0": (guided, {"radius": 2, "eps": 1e-3, "boost": 1.0}),
    "guided-r2b2.0": (guided, {"radius": 2, "eps": 1e-3, "boost": 2.0}),
    "guided-r8e1e-2b1.0": (guided, {"radius": 8, "eps": 1e-2, "boost": 1.0}),  # He et al.'s detail-enhancement style
}
