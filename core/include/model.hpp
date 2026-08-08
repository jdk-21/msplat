#ifndef MODEL_H
#define MODEL_H

#include "metal_tensor.hpp"
#include "ssim.hpp"
#include "input_data.hpp"

int numShBases(int degree);
float psnr(const MTensor& rendered, const MTensor& gt);
float l1_loss(const MTensor& rendered, const MTensor& gt);

struct Model{
  Model(const InputData &inputData, int numCameras,
        int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval,
        int refineEvery, int warmupLength, int resetAlphaEvery, float densifyGradThresh, float densifySizeThresh, int stopScreenSizeAt, float splitScreenSize,
        int maxSteps, bool keepCrs,
        const float* bgColor = nullptr,
        int deformNPoly = 0, int deformNFourier = 0, float deformLr = 0.0f);

  ~Model(){ releaseOptimizers(); }

  void setupOptimizers();
  void releaseOptimizers();

  void schedulersStep(int step);
  int getDownscaleFactor(int step);
  void afterTrain(int step);
  void save(const std::string &filename, int step);
  void savePly(const std::string &filename, int step);
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
  void fullIteration(Camera& cam, int step, MTensor &gt, float ssimWeight);
  MTensor render(Camera& cam, int step);

  MTensor means;
  MTensor scales;
  MTensor quats;
  MTensor featuresDc;
  MTensor featuresRest;
  MTensor opacities;

  // ── 4D (Phase 2) ──────────────────────────────────────────────────────────
  // Trajectory coefficients, (N, 3*deformBasisCount()). Empty for a static run.
  MTensor deform;
  int deformNPoly = 0;      // polynomial order; 0 disables the whole 4D path
  int deformNFourier = 0;   // Fourier order
  float deformLr = 0.0f;    // Adam lr for the trajectory group
  bool is4D() const { return deformNPoly > 0 || deformNFourier > 0; }
  int deformBasisCount() const { return deformNPoly + 2 * deformNFourier; }
  int deformStride() const { return 3 * deformBasisCount(); }
  // Evaluates mu(t) into means_t and returns it; returns the canonical means
  // for a static model, so callers can use the result unconditionally.
  MTensor& deformedMeans(float time);
  MTensor means_t;          // scratch holding mu(t) for the current camera

  static constexpr int N_ADAM_GROUPS = 7;  // 6 static + 1 trajectory group
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
