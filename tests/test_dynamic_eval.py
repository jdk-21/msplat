"""Tests for the masked (dynamic-content) evaluation.

Pure numpy — no GPU, no dataset, no build artefacts. These run even in the
LFS-less checkout where every dataset-backed test in test_msplat.py fails.
"""

import numpy as np
import pytest


def _moving_square(t_frames=16, h=48, w=64, size=10, noise=0.004, seed=0):
    """A bright square crossing a fixed noisy background. Returns (frames, boxes)."""
    rng = np.random.default_rng(seed)
    bg = rng.random((h, w, 3)) * 0.5 + 0.25
    frames = np.empty((t_frames, h, w, 3))
    boxes = []
    for t in range(t_frames):
        f = bg.copy()
        x = 4 + int((w - size - 8) * t / (t_frames - 1))
        y = h // 2 - size // 2
        f[y:y + size, x:x + size] = [0.95, 0.1, 0.1]
        boxes.append((y, x))
        frames[t] = np.clip(f + rng.normal(0, noise, f.shape), 0, 1)
    return frames, boxes


def _truth_union(frames, boxes, size=10):
    truth = np.zeros(frames.shape[1:3], bool)
    for y, x in boxes:
        truth[y:y + size, x:x + size] = True
    return truth


def test_dynamic_mask_finds_the_moving_object():
    from msplat.dynamic_eval import dynamic_mask

    frames, boxes = _moving_square()
    mask, info = dynamic_mask(frames, mode="union")
    truth = _truth_union(frames, boxes)

    # Every truly moving pixel is covered, and the mask stays well short of the
    # whole image — which is the entire point of masking.
    assert (mask & truth).sum() == truth.sum()
    assert 0.0 < info["coverage"] < 0.9
    assert info["num_frames"] == len(frames)


def test_frame_mode_is_more_selective_than_union():
    from msplat.dynamic_eval import dynamic_mask

    frames, _ = _moving_square()
    _, per_frame = dynamic_mask(frames, mode="frame")
    _, union = dynamic_mask(frames, mode="union")
    assert per_frame["coverage"] < union["coverage"]


def test_temporal_radius_widens_the_frame_mask():
    from msplat.dynamic_eval import dynamic_mask

    frames, _ = _moving_square()
    _, tight = dynamic_mask(frames, mode="frame", temporal_radius=0)
    _, wide = dynamic_mask(frames, mode="frame", temporal_radius=3)
    assert wide["coverage"] > tight["coverage"]


def test_dynamic_mask_is_empty_on_a_still_sequence():
    from msplat.dynamic_eval import dynamic_mask

    rng = np.random.default_rng(1)
    base = rng.random((1, 32, 32, 3)) * 0.6 + 0.2
    still = np.clip(np.tile(base, (12, 1, 1, 1))
                    + rng.normal(0, 0.002, (12, 32, 32, 3)), 0, 1)
    _, info = dynamic_mask(still, mode="union")
    assert info["coverage"] < 0.02


def test_dynamic_mask_rejects_too_few_frames():
    from msplat.dynamic_eval import dynamic_mask

    with pytest.raises(ValueError):
        dynamic_mask(np.zeros((2, 8, 8, 3)))


def test_dynamic_mask_rejects_a_bad_shape():
    from msplat.dynamic_eval import dynamic_mask

    with pytest.raises(ValueError):
        dynamic_mask(np.zeros((5, 8, 8)))


def test_masked_metrics_prefer_the_model_that_gets_the_motion_right():
    """The reason this module exists: a global metric can rank these backwards."""
    from msplat.dynamic_eval import dynamic_mask, masked_psnr, masked_ssim

    frames, boxes = _moving_square()
    mask, _ = dynamic_mask(frames, mode="union")
    rng = np.random.default_rng(3)

    wrong_motion = frames.copy()      # background perfect, subject flat grey
    wrong_static = frames.copy()      # subject perfect, background noisy
    for t, (y, x) in enumerate(boxes):
        wrong_motion[t, y:y + 10, x:x + 10] = 0.5
        wrong_static[t] = np.clip(
            wrong_static[t] + rng.normal(0, 0.05, frames.shape[1:]), 0, 1)
        wrong_static[t, y:y + 10, x:x + 10] = frames[t, y:y + 10, x:x + 10]

    n = len(frames)
    dyn_a = np.mean([masked_psnr(wrong_motion[t], frames[t], mask) for t in range(n)])
    dyn_b = np.mean([masked_psnr(wrong_static[t], frames[t], mask) for t in range(n)])
    assert dyn_b > dyn_a + 3.0

    ssim_a = np.mean([masked_ssim(wrong_motion[t], frames[t], mask) for t in range(n)])
    ssim_b = np.mean([masked_ssim(wrong_static[t], frames[t], mask) for t in range(n)])
    assert ssim_b > ssim_a


def test_masked_psnr_matches_unmasked_when_the_mask_is_all_true():
    from msplat.dynamic_eval import masked_psnr

    rng = np.random.default_rng(5)
    a = rng.random((16, 16, 3))
    b = np.clip(a + rng.normal(0, 0.05, a.shape), 0, 1)
    assert masked_psnr(a, b) == pytest.approx(
        masked_psnr(a, b, np.ones(a.shape[:2], bool)))


def test_identical_images_score_perfectly():
    from msplat.dynamic_eval import masked_psnr, masked_ssim

    rng = np.random.default_rng(7)
    a = rng.random((24, 24, 3))
    assert np.isinf(masked_psnr(a, a))
    assert masked_ssim(a, a) == pytest.approx(1.0, abs=1e-9)


def test_masked_metrics_return_nan_on_an_empty_mask():
    from msplat.dynamic_eval import masked_psnr, masked_ssim

    a, b = np.zeros((8, 8, 3)), np.ones((8, 8, 3))
    empty = np.zeros((8, 8), bool)
    assert np.isnan(masked_psnr(a, b, empty))
    assert np.isnan(masked_ssim(a, b, empty))


def test_group_by_camera_prefers_cam_id_and_falls_back_to_pose():
    from msplat.dynamic_eval import group_by_camera

    eye = np.eye(4, dtype=np.float32)
    with_ids = [{"cam_id": i % 3, "time": 0.0, "pose": eye} for i in range(6)]
    assert group_by_camera(with_ids) == [[0, 3], [1, 4], [2, 5]]

    # cam_id == -1 means the loader does not know it; identical extrinsics then
    # identify a rigidly mounted camera, the same fallback attachFlowToCameras
    # uses. Mixing two cameras into one background model would make the whole
    # stereo baseline look dynamic.
    without = [{"cam_id": -1, "time": 0.0, "pose": eye * (1 if i % 2 else 2)}
               for i in range(6)]
    assert group_by_camera(without) == [[0, 2, 4], [1, 3, 5]]


def test_dilate_and_erode_do_not_wrap_around_the_border():
    from msplat.dynamic_eval import dilate, erode

    m = np.zeros((9, 9), bool)
    m[0, 0] = True
    grown = dilate(m, 1)
    assert grown[:2, :2].all()
    assert not grown[-1, -1]          # np.roll would have leaked to here
    assert not erode(grown, 1)[1, 1]  # erosion treats outside as solid
