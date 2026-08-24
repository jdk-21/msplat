"""msplat test suite."""

import pytest
import numpy as np
import tempfile
import os
import json
import sys
import types

GARDEN = os.path.join(os.path.dirname(__file__), "..", "datasets", "mipnerf360", "garden")
HAS_GARDEN = os.path.isdir(GARDEN)


# ── Import tests ─────────────────────────────────────────────────────────────


def test_import():
    import msplat
    assert hasattr(msplat, "GaussianTrainer")
    assert hasattr(msplat, "TrainingConfig")
    assert hasattr(msplat, "Dataset")
    assert hasattr(msplat, "load_dataset")


def test_training_config_defaults():
    from msplat import TrainingConfig

    cfg = TrainingConfig()
    assert cfg.iterations == 30000
    assert cfg.sh_degree == 3
    assert cfg.ssim_weight == pytest.approx(0.2)
    assert cfg.refine_every == 100
    assert cfg.warmup_length == 500


def test_training_config_custom():
    from msplat import TrainingConfig

    cfg = TrainingConfig(iterations=100, sh_degree=1, ssim_weight=0.0)
    assert cfg.iterations == 100
    assert cfg.sh_degree == 1
    assert cfg.ssim_weight == 0.0


def test_training_config_mutable():
    from msplat import TrainingConfig

    cfg = TrainingConfig()
    cfg.iterations = 500
    assert cfg.iterations == 500


def test_cli_checkpoint_writes_provenance_sidecar_without_export_frames(tmp_path, monkeypatch):
    """The post-training checkpoint branch writes its sidecar before the optional frame export path."""
    from msplat import cli

    args = types.SimpleNamespace(
        input=str(tmp_path / "dataset"),
        output=str(tmp_path / "scene.ply"),
        checkpoint=str(tmp_path / "scene.msplat"),
        num_iters=1,
        downscale_factor=1.0,
        num_downscales=0,
        resolution_schedule=3000,
        sh_degree=0,
        sh_degree_interval=1000,
        ssim_weight=0.2,
        refine_every=100,
        warmup_length=500,
        reset_alpha_every=30,
        densify_grad_thresh=0.0002,
        densify_size_thresh=0.01,
        stop_screen_size_at=4000,
        split_screen_size=0.05,
        keep_crs=False,
        save_every=-1,
        eval=False,
        test_every=8,
        deform_n_poly=0,
        deform_n_fourier=0,
        deform_lr=0.001,
        deform_rot_n_poly=0,
        deform_rot_n_fourier=0,
        deform_rot_lr=0.0001,
        deform_temporal=False,
        deform_temp_lr=0.01,
        flow=False,
        flow_weight=0.03,
        flow_min_coverage=0.1,
        rigid=False,
        rigid_weight=0.5,
        rigid_beta=100.0,
        rigid_k=20,
        white_background=False,
        export_frames=0,
        export_frames_dir="",
        export_frames_t0=0.0,
        export_frames_t1=1.0,
        export_frames_full_sh=False,
    )

    class FakeTrainingConfig:
        def __init__(self, **kwargs):
            for name, value in kwargs.items():
                setattr(self, name, value)
            self.bg_color = [0.6130, 0.0101, 0.3984]

    class FakeDataset:
        num_train = 1
        num_test = 0

        def __init__(self, *args, **kwargs):
            pass

    class FakeTrainer:
        def __init__(self, *args):
            pass

        def train(self, *args, **kwargs):
            pass

        def export_ply(self, path):
            pass

        def save_checkpoint(self, path):
            with open(path, "wb") as checkpoint:
                checkpoint.write(b"checkpoint")

    fake_tyro = types.ModuleType("tyro")
    fake_tyro.cli = lambda _: args
    fake_msplat = types.ModuleType("msplat")
    fake_msplat.TrainingConfig = FakeTrainingConfig
    fake_msplat.Dataset = FakeDataset
    fake_msplat.GaussianTrainer = FakeTrainer
    fake_msplat.sync = lambda: None
    fake_msplat.cleanup = lambda: None
    monkeypatch.setitem(sys.modules, "tyro", fake_tyro)
    monkeypatch.setitem(sys.modules, "msplat", fake_msplat)

    cli.main()

    sidecar = tmp_path / "scene.msplat.json"
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["checkpoint"] == os.path.abspath(args.checkpoint)


# ── Dataset tests ────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_load_dataset():
    from msplat import Dataset

    ds = Dataset(GARDEN, downscale_factor=4.0, eval_mode=True, test_every=8)
    assert ds.num_train > 0
    assert ds.num_test > 0
    assert ds.num_train + ds.num_test > 100


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_load_dataset_no_eval():
    from msplat import Dataset

    ds = Dataset(GARDEN, downscale_factor=4.0, eval_mode=False)
    assert ds.num_train > 0
    assert ds.num_test == 0


# ── Training tests ───────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_train_short():
    """Train 50 steps at 4x downscale — verify it runs without error."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=50, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    steps_seen = []
    trainer.train(lambda s: steps_seen.append(s.iteration), callback_every=10)

    assert trainer.iteration == 50
    assert trainer.splat_count > 100000
    assert steps_seen == [10, 20, 30, 40, 50]


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_step_by_step():
    """Manual step loop works."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        stats = trainer.step()

    assert stats.iteration == 10
    assert stats.splat_count > 0
    assert stats.ms_per_step > 0


# ── Render tests ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_render():
    """Render produces valid image array."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer, sync

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        trainer.step()

    img = trainer.render(0)
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.float32
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.shape[0] > 0 and img.shape[1] > 0
    # Values should be in [0, 1] range (approximately)
    assert img.min() >= -0.1
    assert img.max() <= 1.5


# ── Export tests ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_export_ply():
    """PLY export creates a valid file."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        trainer.step()

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name

    try:
        trainer.export_ply(path)
        assert os.path.exists(path)
        size = os.path.getsize(path)
        assert size > 1000  # non-trivial file
    finally:
        os.unlink(path)


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_export_splat():
    """Splat export creates a valid file."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        trainer.step()

    with tempfile.NamedTemporaryFile(suffix=".splat", delete=False) as f:
        path = f.name

    try:
        trainer.export_splat(path)
        assert os.path.exists(path)
        size = os.path.getsize(path)
        assert size > 1000
    finally:
        os.unlink(path)


# ── Eval tests ───────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_evaluate():
    """Evaluation returns valid metrics dict."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0, eval_mode=True, test_every=8)
    cfg = TrainingConfig(iterations=50, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    trainer.train(lambda s: None, callback_every=50)
    metrics = trainer.evaluate()

    assert "psnr" in metrics
    assert "ssim" in metrics
    assert "l1" in metrics
    assert "num_test" in metrics
    assert metrics["num_test"] > 0
    assert metrics["psnr"] > 10  # sanity — should be at least somewhat trained
    assert 0 < metrics["ssim"] < 1
    assert metrics["l1"] > 0


# ── Checkpoint tests ────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_checkpoint_save_load():
    """Save checkpoint, load it, verify state is preserved."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=100, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(50):
        trainer.step()

    splats_at_50 = trainer.splat_count

    with tempfile.NamedTemporaryFile(suffix=".msplat", delete=False) as f:
        ckpt_path = f.name

    try:
        trainer.save_checkpoint(ckpt_path)
        assert os.path.exists(ckpt_path)
        assert os.path.getsize(ckpt_path) > 1000

        # Load into a fresh trainer
        ds2 = Dataset(GARDEN, downscale_factor=4.0)
        cfg2 = TrainingConfig(iterations=100, num_downscales=0)
        trainer2 = GaussianTrainer(ds2, cfg2)
        trainer2.load_checkpoint(ckpt_path)

        assert trainer2.iteration == 50
        assert trainer2.splat_count == splats_at_50
    finally:
        os.unlink(ckpt_path)


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_checkpoint_resume_training():
    """Train 50 → save → load → train 50 more. Verify it completes."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=100, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(50):
        trainer.step()

    with tempfile.NamedTemporaryFile(suffix=".msplat", delete=False) as f:
        ckpt_path = f.name

    try:
        trainer.save_checkpoint(ckpt_path)

        # Resume in a new trainer
        ds2 = Dataset(GARDEN, downscale_factor=4.0)
        cfg2 = TrainingConfig(iterations=100, num_downscales=0)
        trainer2 = GaussianTrainer(ds2, cfg2)
        trainer2.load_checkpoint(ckpt_path)

        for _ in range(50):
            stats = trainer2.step()

        assert trainer2.iteration == 100
        assert stats.splat_count > 0
        assert stats.ms_per_step > 0
    finally:
        os.unlink(ckpt_path)
