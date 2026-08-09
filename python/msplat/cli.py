"""msplat-train CLI entry point."""

import sys


def main():
    try:
        import tyro
    except ImportError:
        print("Install tyro for CLI support: pip install msplat[cli]", file=sys.stderr)
        sys.exit(1)

    from dataclasses import dataclass, field

    @dataclass
    class Args:
        """Train 3D Gaussian Splatting on a dataset."""

        input: str
        """Path to dataset (COLMAP, Nerfstudio, etc.)"""

        output: str = "splat.ply"
        """Output PLY file path"""

        num_iters: int = 30000
        """Number of training iterations"""

        downscale_factor: float = 1.0
        """Image downscale factor"""

        num_downscales: int = 2
        """Number of progressive downscales"""

        resolution_schedule: int = 3000
        """Double resolution every N steps"""

        sh_degree: int = 3
        """Max spherical harmonics degree"""

        sh_degree_interval: int = 1000
        """Steps between SH degree increases"""

        ssim_weight: float = 0.2
        """SSIM loss weight"""

        refine_every: int = 100
        """Densification interval"""

        warmup_length: int = 500
        """Steps before densification starts"""

        reset_alpha_every: int = 30
        """Reset opacity every N refinements"""

        densify_grad_thresh: float = 0.0002
        """Gradient threshold for densification"""

        densify_size_thresh: float = 0.01
        """Size threshold for split vs clone"""

        stop_screen_size_at: int = 4000
        """Stop screen-size split after this step"""

        split_screen_size: float = 0.05
        """Screen-space split threshold"""

        keep_crs: bool = False
        """Keep input coordinate reference system"""

        save_every: int = -1
        """Save every N steps (-1 to disable)"""

        eval: bool = False
        """Evaluate on held-out test views"""

        test_every: int = 8
        """Hold out every Nth image for eval (ignored for datasets with their own split)"""

        deform_n_poly: int = 0
        """4D: polynomial order of the per-Gaussian trajectory (0 = static 3DGS)"""

        deform_n_fourier: int = 0
        """4D: Fourier order of the per-Gaussian trajectory (0 = static 3DGS)"""

        deform_lr: float = 0.001
        """4D: Adam learning rate for the trajectory coefficients"""

        deform_rot_n_poly: int = 0
        """4D: polynomial order of the per-Gaussian rotation trajectory (paper: 3)"""

        deform_rot_n_fourier: int = 0
        """4D: Fourier order of the per-Gaussian rotation trajectory (paper: 3)"""

        deform_rot_lr: float = 0.0001
        """4D: Adam learning rate for the rotation coefficients"""

        deform_temporal: bool = False
        """Phase 4: per-Gaussian temporal envelope on opacity — a Gaussian may exist
        for only part of the sequence instead of having to explain every frame"""

        deform_temp_lr: float = 0.01
        """Phase 4: Adam learning rate for the envelope's centre and width"""

        flow: bool = False
        """Phase 3: render optical flow from the velocity field and apply L_flow.
        Needs ground-truth flow in <input>/flow/<image stem>.flo (see tools/precompute_flow.py)."""

        flow_weight: float = 0.03
        """Phase 3: gamma2 in L = L_color + gamma2*L_flow + gamma3*L_rigid"""

        flow_min_coverage: float = 0.1
        """Phase 3: ignore flow pixels where the model accumulates less alpha than this"""

        rigid: bool = False
        """Phase 3: apply L_rigid — neighbouring Gaussians should move alike"""

        rigid_weight: float = 0.5
        """Phase 3: gamma3, the L_rigid weight"""

        rigid_beta: float = 100.0
        """Phase 3: neighbour weight w_ij = exp(-beta * ||mu_i - mu_j||), in world units"""

        rigid_k: int = 20
        """Phase 3: neighbours per Gaussian for L_rigid"""

        white_background: bool = False
        """Composite transparent source images over white (D-NeRF protocol)"""

        export_frames: int = 0
        """4D: bake this many PLY frames across the time range (0 = off)"""

        export_frames_dir: str = ""
        """4D: directory for the frame sequence (default: <output stem>_frames)"""

        export_frames_t0: float = 0.0
        """4D: first sampled time, normalized to [0,1]"""

        export_frames_t1: float = 1.0
        """4D: last sampled time, normalized to [0,1]"""

        export_frames_full_sh: bool = False
        """4D: keep full SH in the frames (default: DC only, ~4x smaller files)"""

    args = tyro.cli(Args)

    from msplat import TrainingConfig, Dataset, GaussianTrainer, sync, cleanup

    config = TrainingConfig(
        iterations=args.num_iters,
        sh_degree=args.sh_degree,
        sh_degree_interval=args.sh_degree_interval,
        ssim_weight=args.ssim_weight,
        num_downscales=args.num_downscales,
        resolution_schedule=args.resolution_schedule,
        refine_every=args.refine_every,
        warmup_length=args.warmup_length,
        reset_alpha_every=args.reset_alpha_every,
        densify_grad_thresh=args.densify_grad_thresh,
        densify_size_thresh=args.densify_size_thresh,
        stop_screen_size_at=args.stop_screen_size_at,
        split_screen_size=args.split_screen_size,
        keep_crs=args.keep_crs,
        downscale_factor=args.downscale_factor,
        output=args.output,
        save_every=args.save_every,
        deform_n_poly=args.deform_n_poly,
        deform_n_fourier=args.deform_n_fourier,
        deform_lr=args.deform_lr,
        deform_rot_n_poly=args.deform_rot_n_poly,
        deform_rot_n_fourier=args.deform_rot_n_fourier,
        deform_rot_lr=args.deform_rot_lr,
        flow=args.flow,
        flow_weight=args.flow_weight,
        flow_min_coverage=args.flow_min_coverage,
        rigid=args.rigid,
        rigid_weight=args.rigid_weight,
        rigid_beta=args.rigid_beta,
        rigid_k=args.rigid_k,
        deform_temporal=args.deform_temporal,
        deform_temp_lr=args.deform_temp_lr,
    )

    if (args.flow or args.rigid) and args.deform_n_poly == 0 and args.deform_n_fourier == 0:
        print(
            "Error: --flow/--rigid need a trajectory to supervise. "
            "Add --deform-n-poly / --deform-n-fourier.",
            file=sys.stderr,
        )
        sys.exit(2)

    # The rasterizer's background has to match what transparent source pixels
    # were composited over, or the eval PSNR measures the mismatch.
    if args.white_background:
        config.bg_color = [1.0, 1.0, 1.0]

    dataset = Dataset(
        args.input,
        downscale_factor=args.downscale_factor,
        eval_mode=args.eval,
        test_every=args.test_every,
        white_background=args.white_background,
    )
    print(f"Loaded {dataset.num_train} train cameras", end="")
    if args.eval:
        print(f", {dataset.num_test} test cameras")
    else:
        print()

    trainer = GaussianTrainer(dataset, config)

    def on_step(stats):
        line = (
            f"step={stats.iteration:>6}  "
            f"splats={stats.splat_count:>8,}  "
            f"ms={stats.ms_per_step:.1f}"
        )
        if args.flow:
            line += f"  flow_l1={stats.flow_loss:.3f}px"
        if args.rigid:
            line += f"  rigid={stats.rigid_loss:.5f}"
        if args.deform_temporal:
            line += (f"  env={stats.temporal_mean:.3f}"
                     f"  aus={stats.temporal_off_frac * 100:.1f}%")
        print(line)

    trainer.train(on_step, callback_every=100)

    trainer.export_ply(args.output)
    print(f"Saved {args.output}")

    if args.export_frames > 0:
        import os.path

        if args.deform_n_poly == 0 and args.deform_n_fourier == 0:
            print(
                "Warning: --export-frames on a static model — all frames identical.",
                file=sys.stderr,
            )
        frames_dir = args.export_frames_dir or (
            os.path.splitext(args.output)[0] + "_frames"
        )
        trainer.export_ply_sequence(
            frames_dir,
            prefix="frame",
            num_frames=args.export_frames,
            t0=args.export_frames_t0,
            t1=args.export_frames_t1,
            max_sh_bases=-1 if args.export_frames_full_sh else 0,
        )
        print(
            f"Frame sequence in {frames_dir}/ — open https://superspl.at/editor "
            "and drop all frame_*.ply in together to play it back."
        )

    if args.eval:
        metrics = trainer.evaluate()
        print(f"\n=== Evaluation ({metrics['num_test']} test views) ===")
        print(f"  PSNR:  {metrics['psnr']:.4f}")
        print(f"  SSIM:  {metrics['ssim']:.4f}")
        print(f"  L1:    {metrics['l1']:.4f}")
        print(f"  Gaussians: {metrics['num_gaussians']:,}")

    cleanup()


if __name__ == "__main__":
    main()
