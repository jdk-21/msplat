#ifndef MODEL_H
#define MODEL_H

#include "metal_tensor.hpp"
#include "ssim.hpp"
#include "input_data.hpp"
#include "bindings.h"   // DeformOrders

int numShBases(int degree);
float psnr(const MTensor& rendered, const MTensor& gt);
float l1_loss(const MTensor& rendered, const MTensor& gt);

// Everything the 4D/flow extension adds to a training run, kept together so the
// three frontends can hand it through as one value instead of a dozen scalars.
struct DeformConfig {
  DeformOrders ord;             // trajectory orders for means and rotation
  float lr = 0.0f;              // Adam lr for the means trajectory coefficients
  float rotLr = 0.0f;           // Adam lr for the rotation coefficients

  // ── Phase 3 ───────────────────────────────────────────────────────────────
  bool flow = false;            // render optical flow and apply L_flow
  float flowWeight = 0.03f;     // gamma2 in L = L_color + gamma2*L_flow + gamma3*L_rigid
  float flowMinCoverage = 0.1f; // ignore pixels the model leaves (near-)empty
  bool rigid = false;           // apply L_rigid over the kNN graph
  float rigidWeight = 0.5f;     // gamma3
  float rigidBeta = 100.0f;     // w_ij = exp(-beta * ||mu_i - mu_j||)
  int rigidK = 20;              // neighbours per gaussian

  // ── Phase 4 ───────────────────────────────────────────────────────────────
  // Per-gaussian temporal envelope on opacity. Switch it on via ord.temporal;
  // tempLr is its own Adam step size, because centre and width live on a very
  // different scale from the trajectory coefficients.
  float tempLr = 0.0f;

  // L1 shrinkage on the trajectory coefficients, as a proximal step after Adam.
  // The basis starts at tau^1, so every coefficient is pure motion and an exact
  // zero means "this gaussian stands still". On a stage that is the majority
  // case: the room does not move, a small part of it does. Without this, all
  // 3*(nPoly + 2*nFourier) coefficients of a floor gaussian are free to fit
  // per-frame photometric noise, and the static scene jitters — measured on
  // rig4 at 18x the temporal variation of the ground truth, carrying two thirds
  // of the total novel-view error.
  float l1 = 0.0f;

  // Depth supervision and opacity entropy. Independent of the 4D machinery —
  // both work on a purely static model too, which is the point: they constrain
  // geometry, and a sparse rig gets that wrong whether or not it is moving.
  bool depth = false;
  float depthWeight = 0.5f;
  float depthMinCoverage = 0.5f;
  float opacityEntropyWeight = 0.0f;

  bool any4D() const { return ord.any(); }
  bool anyPhase3() const { return flow || rigid; }
  bool temporal() const { return ord.temporal; }
};

struct Model{
  Model(const InputData &inputData, int numCameras,
        int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval,
        int refineEvery, int warmupLength, int resetAlphaEvery, float densifyGradThresh, float densifySizeThresh, int stopScreenSizeAt, float splitScreenSize,
        int maxSteps, bool keepCrs,
        const float* bgColor = nullptr,
        const DeformConfig &deformCfg = DeformConfig{});

  ~Model(){ releaseOptimizers(); }

  void setupOptimizers();
  void releaseOptimizers();

  void schedulersStep(int step);
  int getDownscaleFactor(int step);
  void afterTrain(int step);
  void save(const std::string &filename, int step);
  void savePly(const std::string &filename, int step);
  // Same as savePly, but with the means evaluated at normalized `time` in
  // [0,1]. A static model ignores the time and writes the canonical means.
  void savePlyAt(const std::string &filename, int step, float time, int maxShBases = -1);
  // Bakes numFrames PLYs sampled over [t0, t1] into dir as <prefix>_0000.ply …
  // — a frame sequence any 3DGS viewer can play back as an animation.
  void savePlySequence(const std::string &dir, const std::string &prefix, int step,
                       int numFrames, float t0 = 0.0f, float t1 = 1.0f, int maxShBases = -1);
  void saveSplat(const std::string &filename);
  int loadPly(const std::string &filename);
  void saveCheckpoint(const std::string &filename, int step);
  int loadCheckpoint(const std::string &filename);
  struct CamSetup {
    float fx, fy, cx, cy;
    int height, width, degree, degreesToUse;
    std::tuple<int,int,int> tileBounds;
    float cam_pos[3];
  };
  CamSetup prepareCam(Camera& cam, int step);
  // nextCam is the frame this camera's ground-truth optical flow points at.
  // Pass nullptr (or leave it out) to skip the flow pass for this step.
  void fullIteration(Camera& cam, int step, MTensor &gt, float ssimWeight,
                     Camera* nextCam = nullptr);
  MTensor render(Camera& cam, int step);
  // The optical flow this model predicts from cam to nextCam over dt, as an
  // (H, W, 2) image in pixels per frame interval. Runs a colour forward pass
  // first because the flow compositing reuses its sorted tile lists.
  MTensor renderFlow(Camera& cam, Camera& nextCam, int step, float dt);

  MTensor means;
  MTensor scales;
  MTensor quats;
  MTensor featuresDc;
  MTensor featuresRest;
  MTensor opacities;

  // ── 4D (Phase 2) / flow splatting (Phase 3) ───────────────────────────────
  // One combined coefficient block per gaussian, (N, deformStride()):
  // [ deformBasisCount() float3 means blocks | rotBasisCount() float4 rotation
  // blocks ]. Empty for a static run. Keeping it in a single tensor is what lets
  // densification carry the whole 4D state as one extra buffer.
  MTensor deform;
  DeformConfig dcfg;
  bool is4D() const { return dcfg.ord.any(); }
  bool hasRot() const { return dcfg.ord.hasRot(); }
  int deformBasisCount() const { return dcfg.ord.basisM(); }
  int rotBasisCount() const { return dcfg.ord.basisQ(); }
  int deformStride() const { return dcfg.ord.stride(); }
  // Evaluates mu(t) into means_t (and q(t) into quats_t) and returns the means;
  // returns the canonical means for a static model, so callers can use the
  // result unconditionally. deformedQuats() likewise falls back to quats.
  MTensor& deformedMeans(float time);
  MTensor& deformedQuats() { return hasRot() ? quats_t : quats; }
  MTensor means_t;          // scratch holding mu(t) for the current camera
  MTensor quats_t;          // scratch holding q(t)

  // Phase 3 scratch. velocity/v_vel/v_mu_extra/flow2d/v_flow2d are per-gaussian
  // and reallocated by refreshViews(); neighbors is the kNN graph over the
  // canonical means, rebuilt after every densification.
  MTensor velocity, v_vel, v_mu_extra, flow2d, v_flow2d;
  MTensor neighbors;
  void rebuildNeighbors();
  // Phase 4: w(tau) per gaussian and its gradient. Also per-gaussian scratch
  // from refreshViews(). Undefined when the envelope is off, which is what
  // switches the whole feature off downstream.
  MTensor temporal_w, v_temporal_w;
  // Mean envelope weight over the last evaluated timestamp, and the share of
  // gaussians the envelope has effectively switched off (w < 0.01). The paper's
  // efficiency claim lives in that second number.
  float lastTemporalMean = 1.0f, lastTemporalOffFrac = 0.0f;
  float lastFlowLoss = 0.0f, lastRigidLoss = 0.0f;
  // Steps that actually ran a flow pass, and steps that had to skip it because
  // the progressive downscale did not match the ground-truth flow resolution.
  long flowSteps = 0, flowSkippedSteps = 0;
  // Mean L1 depth error over the supervised pixels (in the scaled world, so
  // comparable across scenes only after undoing InputData::scale) and the mean
  // binary entropy of the opacities in nats. The entropy is the number to watch
  // for fog: ln(2) = 0.69 is a model that has committed to nothing.
  float lastDepthLoss = 0.0f, lastOpacityEntropy = 0.0f;
  // Steps that ran a depth pass, and steps that had ground truth but could not
  // use it — progressive downscaling, or a missing/mismatched .dpt file.
  long depthSteps = 0, depthSkippedSteps = 0;

  static constexpr int N_ADAM_GROUPS = 7;  // 6 static + 1 combined 4D group
  MTensor adam_exp_avg[N_ADAM_GROUPS];
  MTensor adam_exp_avg_sq[N_ADAM_GROUPS];
  int adam_step_count = 0;
  float adam_lr[N_ADAM_GROUPS] = {};
  float adam_beta1 = 0.9f, adam_beta2 = 0.999f, adam_eps = 1e-8f;
  float means_lr_init = 0, means_lr_final = 0;

  MTensor means_buf, scales_buf, quats_buf, featuresDc_buf, featuresRest_buf, opacities_buf;
  MTensor deform_buf;
  MTensor adam_exp_avg_buf[N_ADAM_GROUPS], adam_exp_avg_sq_buf[N_ADAM_GROUPS];
  int num_active = 0, buf_capacity = 0;
  void refreshViews();
  void ensureCapacity(int needed);

  MTensor densify_split_flag, densify_dup_flag;
  MTensor densify_split_prefix, densify_dup_prefix;
  MTensor densify_keep_flag, densify_keep_prefix;
  MTensor densify_block_totals;
  MTensor densify_compact_scratch;
  MTensor densify_random_samples;

  MTensor radii;
  int lastHeight;
  int lastWidth;

  MTensor xysGradNorm;
  MTensor visCounts;
  MTensor max2DSize;

  MTensor backgroundColor;
  MTensor window2d;  // SSIM window (11,11) f32

  int numCameras;
  int numDownscales;
  int resolutionSchedule;
  int shDegree;
  int shDegreeInterval;
  int refineEvery;
  int warmupLength;
  int resetAlphaEvery;
  int stopSplitAt;
  float densifyGradThresh;
  float densifySizeThresh;
  int stopScreenSizeAt;
  float splitScreenSize;
  int maxSteps;
  bool keepCrs;

  float scale;
  float translation[3] = {};
  // 1.1 * max camera distance from the (already centered) origin — the same
  // convention as reference 3DGS's cameras_extent. Used to scale the
  // huge-gaussian cull threshold.
  float sceneExtent = 1.0f;
};

#endif
