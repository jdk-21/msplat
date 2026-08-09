#pragma once

// Swift-compatible C++ API for msplat.
// Designed for Swift 5.9+ C++ interop — no std::tuple,
// no std::unordered_map, no templates in the public interface.
// Internal types hidden via PIMPL.

#include <cstdint>
#include <memory>
#include <string>

namespace msplat {

// ── Config ──────────────────────────────────────────────────────────────────

struct Config {
    int iterations = 30000;
    int shDegree = 3;
    int shDegreeInterval = 1000;
    float ssimWeight = 0.2f;
    int numDownscales = 2;
    int resolutionSchedule = 3000;
    int refineEvery = 100;
    int warmupLength = 500;
    int resetAlphaEvery = 30;
    float densifyGradThresh = 0.0002f;
    float densifySizeThresh = 0.01f;
    int stopScreenSizeAt = 4000;
    float splitScreenSize = 0.05f;
    bool keepCrs = false;
    float downscaleFactor = 1.0f;
    float bgColor[3] = {0.6130f, 0.0101f, 0.3984f};  // magenta — high contrast for debugging

    // ── 4D (Phase 2) ────────────────────────────────────────────────────────
    // Trajectory orders for the time-varying means. Both 0 = static 3DGS,
    // bit-for-bit the old behaviour. The defaults below are only applied when
    // the caller opts in.
    int deformNPoly = 0;        // polynomial order N
    int deformNFourier = 0;     // Fourier order L
    float deformLr = 0.001f;    // Adam lr for the trajectory coefficients
    // Time-varying rotation. Paper uses N=3, L=3 with the scales left constant.
    int deformRotNPoly = 0;
    int deformRotNFourier = 0;
    float deformRotLr = 0.0001f;
    // Temporal envelope (Phase 4): w(tau) = exp(-(tau-m)^2 * softplus(s)) on
    // opacity, so a gaussian need not explain every frame of the sequence.
    bool deformTemporal = false;
    float deformTempLr = 0.01f;

    // ── Flow splatting (Phase 3) ────────────────────────────────────────────
    // L = L_color + flowWeight * L_flow + rigidWeight * L_rigid.
    // flow needs ground-truth optical flow in <dataset>/flow/<stem>.flo.
    bool flow = false;
    float flowWeight = 0.03f;        // gamma2
    float flowMinCoverage = 0.1f;    // skip pixels the model leaves near-empty
    bool rigid = false;
    float rigidWeight = 0.5f;        // gamma3
    float rigidBeta = 100.0f;        // w_ij = exp(-beta * ||mu_i - mu_j||)
    int rigidK = 20;                 // neighbours per gaussian
    // Composite transparent source images over white instead of black. The
    // published D-NeRF protocol uses white; must match bgColor to evaluate fairly.
    bool whiteBackground = false;
};

// ── Stats ───────────────────────────────────────────────────────────────────

struct Stats {
    int iteration = 0;
    int splatCount = 0;
    float msPerStep = 0.0f;
};

struct EvalMetrics {
    float psnr = 0.0f;
    float ssim = 0.0f;
    float l1 = 0.0f;
    int numTest = 0;
    int numGaussians = 0;
};

// ── PixelBuffer ─────────────────────────────────────────────────────────────

/// Rendered image data. Owns its pixel buffer.
struct PixelBuffer {
    float* data = nullptr;   // RGB float32, HWC layout
    int width = 0;
    int height = 0;

    PixelBuffer() = default;
    PixelBuffer(float* d, int w, int h) : data(d), width(w), height(h) {}
    PixelBuffer(const PixelBuffer&) = delete;
    PixelBuffer& operator=(const PixelBuffer&) = delete;
    PixelBuffer(PixelBuffer&& o) : data(o.data), width(o.width), height(o.height) {
        o.data = nullptr;
    }
    PixelBuffer& operator=(PixelBuffer&& o) {
        if (this != &o) {
            free(data);
            data = o.data; width = o.width; height = o.height;
            o.data = nullptr;
        }
        return *this;
    }
    ~PixelBuffer() { free(data); }
};

// ── Dataset ─────────────────────────────────────────────────────────────────

class Dataset {
public:
    Dataset(const std::string& path, float downscaleFactor,
            bool evalMode, int testEvery, bool whiteBackground = false);
    ~Dataset();

    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;
    Dataset(Dataset&&) noexcept;
    Dataset& operator=(Dataset&&) noexcept;

    int numTrain() const;
    int numTest() const;
    void cameraPose(int index, float camToWorld[16]) const;

    // Opaque handle for Trainer
    void* _handle() const;

    struct Impl;
private:
    std::unique_ptr<Impl> impl;
};

// ── Trainer ─────────────────────────────────────────────────────────────────

class Trainer {
public:
    Trainer(Dataset& dataset, const Config& config);
    ~Trainer();

    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;

    /// Run one training step. Returns stats.
    Stats step();

    /// Run N steps (from current iteration to config.iterations).
    /// Calls callback every callbackEvery steps with current Stats.
    /// Use callbackEvery=0 to disable callbacks.
    void train(int callbackEvery);

    /// Evaluate on held-out test cameras.
    EvalMetrics evaluate();

    /// Render a camera view. Caller owns the returned PixelBuffer.
    PixelBuffer render(int cameraIndex, bool useTest);

    /// Render from an arbitrary camera-to-world pose (4x4 row-major, OpenGL convention).
    /// Uses intrinsics from the given reference camera.
    PixelBuffer renderFromPose(const float camToWorld[16], int refCameraIndex);

    /// Render from pose directly into a caller-provided RGBA uint8 buffer.
    /// outRGBA must hold width*height*4 bytes. Avoids intermediate float allocation.
    /// Call with outRGBA=nullptr to query dimensions only.
    void renderFromPoseToBuffer(const float camToWorld[16], int refCameraIndex,
                            uint8_t* outRGBA, int* outWidth, int* outHeight);

    /// Export scene to PLY format.
    void exportPly(const std::string& path);

    /// Export a 4D frame sequence: numFrames PLYs sampled over the normalized
    /// time range [t0, t1], written to dir as <prefix>_0000.ply … Feed the
    /// folder to a viewer that plays PLY sequences.
    /// maxShBases: -1 keeps all SH, 0 writes DC only (much smaller previews).
    void exportPlySequence(const std::string& dir, const std::string& prefix,
                           int numFrames, float t0 = 0.0f, float t1 = 1.0f,
                           int maxShBases = -1);

    /// Export scene to .splat format.
    void exportSplat(const std::string& path);

    /// Save full training state (params + optimizer) for resume.
    void saveCheckpoint(const std::string& path);

    /// Load checkpoint and resume training. Returns the saved iteration.
    int loadCheckpoint(const std::string& path);

    int splatCount() const;
    int iteration() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// ── Lifecycle ───────────────────────────────────────────────────────────────

void sync();
void cleanup();

} // namespace msplat
