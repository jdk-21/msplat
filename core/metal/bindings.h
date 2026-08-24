#ifndef MSPLAT_BINDINGS_H
#define MSPLAT_BINDINGS_H

#include <tuple>
#include "metal_tensor.hpp"

// Release all cached GPU tensors (call before exit to prevent GPU memory leak)
void cleanup_msplat_metal();

// Returns the Metal device used by the msplat context (void* in C++, id<MTLDevice> in ObjC++)
#ifdef __OBJC__
id<MTLDevice> msplat_device();
#else
void* msplat_device();
#endif

// GPU tensor allocation (callable from C++ — delegates to Metal device)
MTensor gpu_zeros(std::vector<int64_t> shape, DType dtype);
MTensor gpu_empty(std::vector<int64_t> shape, DType dtype);

// Commit current command buffer (non-blocking)
void msplat_commit();

// Synchronize (commit + wait for completion)
void msplat_gpu_sync();

// GPU timing — non-invasive, uses completion handlers on committed CBs
void msplat_enable_gpu_timing(bool enable);
// Drains accumulated GPU times (ms per CB) into the provided vector. Thread-safe.
void msplat_drain_gpu_times(std::vector<double>& out);
// Drains per-stage GPU times. stage_times must be an array of N_STAGES vectors.
void msplat_drain_stage_times(std::vector<double> stage_times[], int max_stages, int& n_stages,
                              const char** stage_names);

// Render-only forward pass (no loss computation)
// Returns: out_img (H, W, 3) as MTensor
MTensor msplat_render(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background
);

// Fused forward + backward + Adam + grad_stats in one encoder
// Returns: (radii [N], loss_value float)
std::tuple<MTensor, float> msplat_train_step(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background,
    MTensor &gt, MTensor &window2d, float ssim_weight,
    float loss_inv_n, int features_rest_bases,
    int num_adam_groups,
    MTensor adam_params[], MTensor adam_exp_avg[], MTensor adam_exp_avg_sq[],
    float adam_step_sizes[], float adam_bc2_sqrts[],
    float adam_beta1, float adam_beta2, float adam_eps,
    MTensor &vis_counts, MTensor &xys_grad_norm, MTensor &max_2d_size,
    float inv_max_dim
);

int msplat_densify(
    int N, int buf_capacity,
    float grad_thresh, float size_thresh, float screen_thresh, int check_screen,
    float cull_alpha_thresh, float cull_scale_thresh, float cull_screen_size, int check_huge,
    MTensor &xys_grad_norm, MTensor &vis_counts, MTensor &max_2d_size,
    float half_max_dim,
    MTensor &means_buf, MTensor &scales_buf, MTensor &quats_buf,
    MTensor &featuresDc_buf, MTensor &featuresRest_buf, MTensor &opacities_buf,
    int fr_stride,
    MTensor adam_exp_avg_buf[], MTensor adam_exp_avg_sq_buf[],
    MTensor &split_flag, MTensor &dup_flag,
    MTensor &split_prefix, MTensor &dup_prefix,
    MTensor &keep_flag, MTensor &keep_prefix,
    MTensor &block_totals, MTensor &compact_scratch,
    MTensor &random_samples,
    // 4D trajectory coefficients (Phase 2). df_stride == 0 → static run; the
    // deform_buf argument is then ignored.
    MTensor &deform_buf, int df_stride
);

// ── 4D deformation (Phase 2/3) ──────────────────────────────────────────────
// mu(t) = mu + SUM_n a_n*tau^n + SUM_l [b_l*cos(2*pi*l*tau) + c_l*sin(2*pi*l*tau)]
// q(t)  = q  + the same expansion with its own orders and 4-vector coefficients
// All coefficients live in ONE (N, 3*Bm + 4*Bq) tensor; tau is the frame time
// recentred on the middle of the sequence.

// The four trajectory orders, passed around as a unit.
struct DeformOrders {
    int nPoly = 0, nFourier = 0;      // means trajectory
    int qPoly = 0, qFourier = 0;      // rotation trajectory
    // Per-gaussian temporal envelope (Phase 4): two more scalars per gaussian,
    // a centre and a width, appended after the trajectory coefficients.
    //   w(tau) = exp(-(tau - m)^2 * softplus(s))
    // The envelope multiplies opacity, so a gaussian can be present for only
    // part of the sequence instead of having to explain every frame. Zero is a
    // usable initialisation by construction: tau is already centred on the
    // middle of the take, so m = 0 is the middle, and softplus(0) = 0.69 gives a
    // wide, nearly flat envelope whose gradient is alive from the first step.
    bool temporal = false;
    int basisM() const { return nPoly + 2 * nFourier; }
    int basisQ() const { return qPoly + 2 * qFourier; }
    int nTemporal() const { return temporal ? 2 : 0; }
    int stride() const { return 3 * basisM() + 4 * basisQ() + nTemporal(); }
    bool any() const { return stride() > 0; }
    bool hasRot() const { return basisQ() > 0; }
    // Offset of the envelope pair inside one gaussian's block.
    int temporalOffset() const { return 3 * basisM() + 4 * basisQ(); }
};

// Writes mu(t) into means_t and q(t) into quats_t, to be passed to
// msplat_train_step/msplat_render in place of the canonical means/quats.
// Also computes the temporal envelope into temporal_w and publishes it, so that
// every subsequent render/train step for this timestamp folds it into opacity.
// Pass an undefined temporal_w (or ord.temporal == false) to switch it off.
void msplat_deform_forward(
    int num_points, MTensor &means, MTensor &quats, MTensor &deform,
    float tau, DeformOrders ord, MTensor &means_t, MTensor &quats_t,
    MTensor &temporal_w
);

// Currently published envelope, or nullptr. Only for assertions/debugging.
MTensor *msplat_temporal_envelope();

// Extracts d L/d w from the opacity gradient the rasterizer produced under an
// active envelope. Run it after the train step and before the deform backward
// that consumes v_temporal_w.
void msplat_temporal_opacity_fixup(
    int num_points, MTensor &opacities, MTensor &temporal_w,
    MTensor &v_opacity, MTensor &v_temporal_w
);

// d L / d(raw opacity) from the last train step — needed by the fixup above.
MTensor& msplat_train_v_opacity();

// d L / d mu(t) and d L / d q(t) from the last train step — the inputs to the
// coefficient chain rule.
MTensor& msplat_train_v_mean3d();
MTensor& msplat_train_v_quat();
// Projected radii from the last forward pass; <= 0 means the gaussian was culled.
MTensor& msplat_forward_radii();

// Chain-rules the gradients onto the coefficients and applies Adam in one pass.
// The canonical means/quats need no extra work: d mu(t)/d mu = I and
// d q(t)/d q = I, so their existing Adam groups already got the right gradient.
// v_mu_extra and v_vel are the Phase-3 contributions (flow, rigidity); pass
// undefined MTensors to leave them out. Everything lands in ONE Adam step —
// a second step on the same tensor would corrupt the moment estimates.
void msplat_deform_backward_adam(
    int num_points, MTensor &v_mean3d, MTensor &v_quat, MTensor &deform,
    MTensor &exp_avg, MTensor &exp_avg_sq,
    float tau, DeformOrders ord,
    float step_size_means, float step_size_rot,
    float beta1, float beta2, float bc2_sqrt, float eps,
    MTensor &v_mu_extra, MTensor &v_vel,
    MTensor &v_temporal_w, float step_size_temporal
);

// ── Phase 3: flow splatting ─────────────────────────────────────────────────

// v(t) = d mu/dt for every gaussian — the analytic derivative of the trajectory.
// Also clears v_vel/v_mu_extra (GPU-side): it is the first pass of the Phase-3
// block, and L_flow and L_rigid both accumulate into them afterwards.
void msplat_velocity_field(
    int num_points, MTensor &deform, float tau, DeformOrders ord, MTensor &velocity,
    MTensor &v_vel, MTensor &v_mu_extra
);

// Renders the velocity field as optical flow, compares it to gt_flow, and
// accumulates the gradients into v_vel / v_mu_extra. Must run after
// msplat_train_step for the same camera: it reuses that step's sorted tile
// lists, packed gaussian data and final_Ts.
//
// gt_flow is (H, W, 3) — u, v in pixels per frame interval, plus a validity
// flag. projmat_cur/next are the row-major 4x4 proj*view of the two frames.
// Returns the (unweighted) mean L1 flow error for reporting.
float msplat_flow_step(
    int num_points, MTensor &means_t, MTensor &velocity, MTensor &radii,
    MTensor &projmat_cur, MTensor &projmat_next,
    unsigned img_height, unsigned img_width, float cx, float cy,
    float dt, MTensor &gt_flow, float grad_weight, float min_coverage,
    MTensor &v_vel, MTensor &v_mu_extra, MTensor &flow2d, MTensor &v_flow2d
);

// L_rigid over a precomputed kNN graph; accumulates into v_vel.
// Returns the weighted loss for reporting.
float msplat_rigid_step(
    int num_points, MTensor &means, MTensor &velocity, MTensor &neighbors,
    int k, float beta, float weight, MTensor &v_vel
);

// Renders the flow image only (no loss, no gradients) — used by the numerical
// self-checks and by anyone who wants to look at the predicted flow.
MTensor& msplat_flow_render(
    int num_points, MTensor &means_t, MTensor &velocity, MTensor &radii,
    MTensor &projmat_cur, MTensor &projmat_next,
    unsigned img_height, unsigned img_width, float cx, float cy,
    float dt, MTensor &flow2d
);

// ── Phase 4: depth supervision + opacity entropy ────────────────────────────
//
// Both terms run INSIDE the next msplat_train_step rather than after it, so
// they are published here instead of being passed as arguments — the same
// device the temporal envelope uses, and for a stronger reason: the depth
// gradient has to reach v_depth before project_and_sh_backward_kernel turns it
// into a gradient on the means, and that kernel is inside the fused step.
//
// gt_depth is (H, W) float32, metric depth along the view axis in the scaled
// world of the trained scene, 0 meaning "no ground truth for this pixel".
// Pass nullptr (or weight 0) to switch depth supervision off.
//
// Both must be set before EVERY train step, because they persist: a target left
// standing would supervise the next camera with this camera's depth.
void msplat_set_depth_target(MTensor *gt_depth, float weight, float min_coverage);
void msplat_set_opacity_entropy(float weight);

// Last step's depth loss (mean L1 over the supervised pixels, unweighted) and
// mean opacity entropy in nats. Either pointer may be null.
void msplat_last_depth_losses(float *depth_loss, float *entropy);

#endif
