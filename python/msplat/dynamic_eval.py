"""Masked evaluation: score the moving content, not the room around it.

Why this exists
---------------
On a fixed-camera stage the overwhelming majority of every test image is
motionless background. A global PSNR is therefore dominated by how well the
*static* scene was reconstructed, and a 4D model is judged almost entirely on a
job a plain 3D model does at least as well. Measured on the synthetic stage the
static baseline wins on global PSNR while being visibly wrong on the only part
anyone is looking at. Any A/B decision taken on that number is blind.

The fix is to restrict the metric to the pixels that actually move.

Where the mask comes from
-------------------------
Nothing extra has to be captured. A rigidly mounted camera sees the same
background pixel in every frame, so the per-pixel *median over time* of its own
ground-truth frames IS the static background — a moving subject only has to
leave each pixel alone for more than half the sequence, which is exactly what
"something moves through a volume" means. Pixels deviating from that median are
the dynamic ones.

The deviation threshold is per pixel and robust: 1.4826 * MAD over time
estimates the sensor noise of that pixel (dark regions and blown highlights have
very different noise), with an absolute floor for the noise-free synthetic case
where the MAD is exactly zero.

This is deliberately *not* derived from the rendered flow or from the model
under test: the mask has to be a property of the captured data alone, otherwise
two models would be scored on two different pixel sets and the comparison would
be meaningless.

Per-frame vs. union
-------------------
`mode="frame"` (the default) scores the pixels moving around that frame,
`mode="union"` every pixel the action touches anywhere in the sequence.

Union looks like the safer choice — it is threshold-insensitive, and it covers
ghosting, the characteristic failure of a 4D Gaussian model, where density is
left behind at positions the subject has already vacated. It is nevertheless
the wrong default: measured against the exact ground-truth flow of the
synthetic stage, the union of all moving pixels is 30.7% of the image, because
a subject walking across the frame sweeps a third of it. A mask that large
barely differs from no mask at all, which is the very problem this module
exists to solve. The instantaneous moving content is 6.3%.

The default therefore takes the per-frame mask and dilates it *along time* by
`temporal_radius` frames, which reaches the ghost trail without swallowing the
whole path.

How it compares to the true motion field
----------------------------------------
Checked against the exact ground-truth flow of the synthetic stage, the mask
agrees on 60% to 97% of the moving pixels depending on the view. Both kinds of
disagreement are worth knowing about, and neither is a defect for this purpose:

* The mask *misses* the interior of a moving surface of uniform colour, which
  changes position without changing appearance — the aperture problem. Those
  pixels carry almost no image error to measure, so excluding them costs little.
  This is why the agreement is worst (60%) on the view where the actor is a thin
  silhouette and best (97%) where the subject fills more of the frame.
* The mask *adds* cast shadows. Optical flow marks them as static, because the
  floor geometry does not move; the image content there changes anyway, and a 4D
  model has to reproduce it. For an image-quality metric the appearance-based
  answer is the more correct of the two.

Everything here is numpy-only and takes plain arrays, so it runs and is testable
without a GPU or a dataset.
"""

from __future__ import annotations

import numpy as np

# Rec. 601 luma. The mask only needs a scalar per pixel, and luma weights the
# green channel the way the sensor noise actually behaves.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)

# Consistency with a normal distribution: 1.4826 * MAD estimates sigma.
_MAD_TO_SIGMA = 1.4826


# ── kleine Morphologie und Faltung (kein scipy im Abhängigkeitsbaum) ────────

def _sweep(mask, radius, axis, op):
    """Separable box dilation/erosion along one axis, with explicit padding.

    np.roll would wrap the image around; that puts phantom foreground on the
    opposite border, which matters here because a subject often *does* touch
    the frame edge.
    """
    if radius <= 0:
        return mask
    pad = [(radius, radius) if a == axis else (0, 0) for a in range(mask.ndim)]
    # Dilation treats outside as empty, erosion as solid — so neither operation
    # invents or destroys foreground at the image border.
    padded = np.pad(mask, pad, constant_values=(op == "erode"))
    out = None
    n = mask.shape[axis]
    for i in range(2 * radius + 1):
        sl = [slice(None)] * mask.ndim
        sl[axis] = slice(i, i + n)
        win = padded[tuple(sl)]
        out = win if out is None else (out | win if op == "dilate" else out & win)
    return out


def _morph(mask, radius, op, axes=(-2, -1)):
    for axis in axes:
        mask = _sweep(mask, radius, axis % mask.ndim, op)
    return mask


def dilate(mask, radius):
    """Grow the mask by a square structuring element of the given radius."""
    return _morph(mask, radius, "dilate")


def erode(mask, radius):
    """Shrink the mask by a square structuring element of the given radius."""
    return _morph(mask, radius, "erode")


def _gaussian_kernel(sigma=1.5, radius=5):
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def _blur(img, kernel):
    """Separable Gaussian blur of a (H, W) array, edge-padded ('same' output)."""
    r = len(kernel) // 2
    out = img
    for axis in (0, 1):
        pad = [(r, r) if a == axis else (0, 0) for a in range(2)]
        padded = np.pad(out, pad, mode="edge")
        acc = np.zeros_like(img)
        n = img.shape[axis]
        for i, w in enumerate(kernel):
            sl = [slice(None)] * 2
            sl[axis] = slice(i, i + n)
            acc += w * padded[tuple(sl)]
        out = acc
    return out


# ── Maske ───────────────────────────────────────────────────────────────────

def dynamic_mask(frames, *, k_sigma=6.0, abs_thresh=0.02, open_radius=1,
                 dilate_radius=3, temporal_radius=2, mode="frame"):
    """Boolean mask of the moving image content of ONE fixed camera.

    frames : (T, H, W, 3) float in [0, 1], all from the same physical camera,
             in any order — the background model is a median, not a sequence.

    Returns (mask, info). mask is (T, H, W) bool for mode="frame" and (H, W)
    bool for mode="union"; info carries the diagnostics a caller should look at
    before trusting the number, above all `coverage`.

    For reference, on the synthetic stage the exact motion region derived from
    the ground-truth flow is 30.7% of the image as a union over 2.5 s and 6.3%
    per frame — so a per-frame coverage in the single-digit percents is normal
    and does not mean the mask has missed the subject.
    """
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected (T, H, W, 3), got {frames.shape}")
    if len(frames) < 3:
        raise ValueError("a temporal median needs at least 3 frames, "
                         f"got {len(frames)}")
    if mode not in ("union", "frame"):
        raise ValueError(f"unknown mode {mode!r}")

    luma = frames @ _LUMA
    background = np.median(luma, axis=0)
    resid = np.abs(luma - background)

    # Per-pixel robust noise scale. median(|x - median(x)|) is the MAD by
    # definition, so this is one more median over the residual we already have.
    sigma = _MAD_TO_SIGMA * np.median(resid, axis=0)
    thresh = np.maximum(abs_thresh, k_sigma * sigma)

    raw = resid > thresh
    if mode == "union":
        mask = raw.any(axis=0)
    else:
        # Widen each frame's mask along time. A Gaussian model's characteristic
        # failure is ghosting — leftover density where the subject just was —
        # and a mask fitted to the current frame looks away from exactly those
        # pixels. A few frames of slack covers the trail; the full union does
        # not work as a default, because a subject crossing the frame sweeps a
        # third of the image and the metric stops being selective at all.
        mask = _sweep(raw, temporal_radius, 0, "dilate")

    # Opening first: isolated pixels are noise that survived the threshold.
    if open_radius > 0:
        mask = dilate(erode(mask, open_radius), open_radius)

    # Then dilation, and larger — a Gaussian model fails on the *boundary* of a
    # moving subject (soft edges, cast shadow, disocclusion) at least as badly
    # as in its interior, and an exactly-fitted mask would exclude that.
    #
    # A separate closing step to fill the subject's interior was tried and
    # dropped: a moving surface of uniform colour changes nothing inside its own
    # silhouette (the aperture problem), so the raw mask is hollow — but closing
    # it moved coverage by 0.1 to 0.2 percentage points over the whole stage
    # sequence, because this dilation already bridges gaps of twice its radius.
    if dilate_radius > 0:
        mask = dilate(mask, dilate_radius)

    coverage = float(mask.mean())
    info = {
        "coverage": coverage,
        "mode": mode,
        "num_frames": int(len(frames)),
        "median_sigma": float(np.median(sigma)),
        "background": background,
    }
    return mask, info


def group_by_camera(meta, pose_tol=1e-4):
    """Group frame indices by physical camera.

    `meta` is what Dataset.camera_meta() returns. cam_id is authoritative when
    the loader supplies it; otherwise frames are grouped by identical extrinsics,
    because a rigidly mounted camera repeats its pose exactly. This is the same
    fallback attachFlowToCameras uses, and for the same reason: mixing two
    cameras into one background model would make every pixel of the stereo
    baseline look dynamic.
    """
    ids = [int(m["cam_id"]) for m in meta]
    if ids and min(ids) >= 0:
        groups = {}
        for i, cid in enumerate(ids):
            groups.setdefault(cid, []).append(i)
        return [groups[k] for k in sorted(groups)]

    groups = []          # (representative pose, indices)
    for i, m in enumerate(meta):
        pose = np.asarray(m["pose"], dtype=np.float64)
        for ref, idxs in groups:
            if np.max(np.abs(ref - pose)) <= pose_tol:
                idxs.append(i)
                break
        else:
            groups.append((pose, [i]))
    return [idxs for _, idxs in groups]


# ── Metriken ────────────────────────────────────────────────────────────────

def masked_psnr(pred, gt, mask=None, data_range=1.0):
    """PSNR over the masked pixels only (all three channels count)."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    se = (pred - gt) ** 2
    if mask is None:
        mse = float(se.mean())
    else:
        sel = np.asarray(mask, dtype=bool)
        if not sel.any():
            return float("nan")
        mse = float(se[sel].mean())
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim_map(pred, gt, data_range=1.0, sigma=1.5, radius=5):
    """Per-pixel SSIM, 11x11 Gaussian windows, averaged over the channels.

    Split out from masked_ssim because a caller that wants both the global and
    the masked value needs the same map twice, and this is the expensive half.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    kernel = _gaussian_kernel(sigma, radius)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    maps = []
    for c in range(pred.shape[-1]):
        x, y = pred[..., c], gt[..., c]
        mu_x, mu_y = _blur(x, kernel), _blur(y, kernel)
        mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        var_x = _blur(x * x, kernel) - mu_xx
        var_y = _blur(y * y, kernel) - mu_yy
        cov = _blur(x * y, kernel) - mu_xy
        maps.append(((2 * mu_xy + c1) * (2 * cov + c2)) /
                    ((mu_xx + mu_yy + c1) * (var_x + var_y + c2)))
    return np.mean(maps, axis=0)


def mean_over_mask(values, mask=None):
    """Mean of a per-pixel map, restricted to the mask. NaN if the mask is empty."""
    if mask is None:
        return float(values.mean())
    sel = np.asarray(mask, dtype=bool)
    if not sel.any():
        return float("nan")
    return float(values[sel].mean())


def masked_ssim(pred, gt, mask=None, **kw):
    """Mean SSIM over the masked pixels.

    The SSIM map is computed on the whole image and only *averaged* over the
    mask. Cropping the images to the mask first would invent edges at the mask
    boundary and score those instead.

    Absolute values are not comparable to msplat's own Metal SSIM kernel; the
    point is the comparison between the masked and unmasked value, and both come
    from this same function.
    """
    return mean_over_mask(ssim_map(pred, gt, **kw), mask)


# ── Treiber ─────────────────────────────────────────────────────────────────

def evaluate_dynamic(trainer, dataset, *, use_test=True, progress=None, **mask_kw):
    """Score a trained model globally and on the moving content only.

    Renders every frame of the split twice over: once to build the background
    model from the ground truth, once to compare. Only a luma stack is held for
    the first pass, so peak memory stays at one float32 plane per frame.

    Returns a dict with psnr_all/ssim_all, psnr_dyn/ssim_dyn, and the mask
    diagnostics. Both metric pairs come from the numpy implementations above, so
    the *difference* between them is meaningful even though the absolute values
    differ slightly from msplat's own evaluate().
    """
    meta = dataset.camera_meta(use_test)
    if not meta:
        raise RuntimeError("no cameras in this split — load the dataset with "
                           "eval_mode=True")
    groups = group_by_camera(meta)

    totals = {"psnr_all": 0.0, "ssim_all": 0.0, "psnr_dyn": 0.0, "ssim_dyn": 0.0}
    n_scored = 0
    n_skipped = 0
    coverages = []

    for gi, idxs in enumerate(groups):
        if len(idxs) < 3:
            # One or two frames from a camera carry no temporal signal at all;
            # a median of two is meaningless. Report it rather than guessing.
            n_skipped += len(idxs)
            continue

        gts = np.stack([np.asarray(trainer.gt_image(i, use_test)) for i in idxs])
        mask, info = dynamic_mask(gts, **mask_kw)
        coverages.append(info["coverage"])
        del gts

        for k, i in enumerate(idxs):
            gt = np.asarray(trainer.gt_image(i, use_test), dtype=np.float64)
            pred = np.asarray(trainer.render(i, use_test), dtype=np.float64)
            m = mask if mask.ndim == 2 else mask[k]

            # Both metrics are wanted globally and masked, so compute each
            # per-pixel map once and average it twice.
            se = ((pred - gt) ** 2).mean(axis=-1)
            smap = ssim_map(pred, gt)
            for name, sel in (("all", None), ("dyn", m)):
                mse = mean_over_mask(se, sel)
                totals[f"psnr_{name}"] += (float("inf") if mse <= 0.0 else
                                           10.0 * np.log10(1.0 / mse))
                totals[f"ssim_{name}"] += mean_over_mask(smap, sel)
            n_scored += 1
            if progress is not None:
                progress(gi, len(groups), k, len(idxs))

    if n_scored == 0:
        raise RuntimeError("no camera in this split has 3+ frames — the masked "
                           "metric needs a time sequence per camera, which a "
                           "static dataset does not have")

    out = {k: v / n_scored for k, v in totals.items()}
    out["coverage"] = float(np.mean(coverages))
    out["num_frames"] = n_scored
    out["num_cameras"] = len(coverages)
    out["num_skipped"] = n_skipped
    return out
