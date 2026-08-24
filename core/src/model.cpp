#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include "model.hpp"
#include "kdtree_tensor.hpp"
#include "msplat.hpp"
#include "loaders.hpp"

namespace fs = std::filesystem;

static const double C0 = 0.28209479177387814;

int numShBases(int degree){
    switch(degree){
        case 0: return 1;
        case 1: return 4;
        case 2: return 9;
        case 3: return 16;
        default: return 25;
    }
}

// Metrics on CPU MTensor data
float psnr(const MTensor& rendered, const MTensor& gt) {
    int64_t n = rendered.numel();
    const float *r = rendered.data<float>(), *g = gt.data<float>();
    double mse = 0;
    for (int64_t i = 0; i < n; i++) { double d = r[i] - g[i]; mse += d * d; }
    mse /= n;
    return 10.0f * std::log10(1.0 / mse);
}

float l1_loss(const MTensor& rendered, const MTensor& gt) {
    int64_t n = rendered.numel();
    const float *r = rendered.data<float>(), *g = gt.data<float>();
    double sum = 0;
    for (int64_t i = 0; i < n; i++) sum += std::abs(r[i] - g[i]);
    return (float)(sum / n);
}

// Model constructor
Model::Model(const InputData &inputData, int numCameras,
    int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval,
    int refineEvery, int warmupLength, int resetAlphaEvery, float densifyGradThresh, float densifySizeThresh, int stopScreenSizeAt, float splitScreenSize,
    int maxSteps, bool keepCrs,
    const float* bgColor,
    const DeformConfig &deformCfg)
    : dcfg(deformCfg),
      numCameras(numCameras), numDownscales(numDownscales), resolutionSchedule(resolutionSchedule),
      shDegree(shDegree), shDegreeInterval(shDegreeInterval),
      refineEvery(refineEvery), warmupLength(warmupLength), resetAlphaEvery(resetAlphaEvery),
      stopSplitAt(maxSteps / 2), densifyGradThresh(densifyGradThresh), densifySizeThresh(densifySizeThresh),
      stopScreenSizeAt(stopScreenSizeAt), splitScreenSize(splitScreenSize),
      maxSteps(maxSteps), keepCrs(keepCrs) {

    int64_t numPoints = inputData.points.count;
    scale = inputData.scale;
    memcpy(translation, inputData.translation, sizeof(translation));

    // Camera extent in the normalised frame (cameras are centered and scaled
    // by autoScaleAndCenter before the model is built).
    {
        float maxDist = 0.0f;
        for (const auto &cam : inputData.cameras) {
            float x = cam.camToWorld[3], y = cam.camToWorld[7], z = cam.camToWorld[11];
            maxDist = std::max(maxDist, std::sqrt(x*x + y*y + z*z));
        }
        if (maxDist > 0.0f) sceneExtent = 1.1f * maxDist;
    }

    // Means: copy xyz directly to GPU
    means = gpu_empty({numPoints, 3}, DType::Float32);
    memcpy(means.data_ptr(), inputData.points.xyz.data(), numPoints * 3 * sizeof(float));

    // Scales: KD-tree nearest neighbor distances, log'd, repeated 3x
    {
        PointsTensor pt(inputData.points.xyz.data(), numPoints);
        auto sc = pt.scales();  // vector<float> of length numPoints
        scales = gpu_empty({numPoints, 3}, DType::Float32);
        float *sp = scales.data<float>();
        for (int64_t i = 0; i < numPoints; i++) {
            float v = std::log(sc[i]);
            sp[i*3] = sp[i*3+1] = sp[i*3+2] = v;
        }
    }

    // Random quaternions
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        quats = gpu_empty({numPoints, 4}, DType::Float32);
        float *qp = quats.data<float>();
        for (int64_t i = 0; i < numPoints; i++) {
            float u = dist(rng), v = dist(rng), w = dist(rng);
            qp[i*4+0] = std::sqrt(1-u) * std::sin(2*M_PI*v);
            qp[i*4+1] = std::sqrt(1-u) * std::cos(2*M_PI*v);
            qp[i*4+2] = std::sqrt(u) * std::sin(2*M_PI*w);
            qp[i*4+3] = std::sqrt(u) * std::cos(2*M_PI*w);
        }
    }

    // SH features: f_dc = rgb2sh(rgb), f_rest = zeros
    int dimSh = numShBases(shDegree);
    {
        featuresDc = gpu_empty({numPoints, 3}, DType::Float32);
        float *dp = featuresDc.data<float>();
        const uint8_t *rgb = inputData.points.rgb.data();
        for (int64_t i = 0; i < numPoints; i++) {
            for (int c = 0; c < 3; c++)
                dp[i*3+c] = (float)((rgb[i*3+c] / 255.0 - 0.5) / C0);
        }
        featuresRest = gpu_zeros({numPoints, (int64_t)(dimSh - 1), 3}, DType::Float32);
    }

    // Opacities: logit(0.1) = log(0.1/0.9)
    {
        float logit01 = std::log(0.1f / 0.9f);
        opacities = gpu_empty({numPoints, 1}, DType::Float32);
        float *op = opacities.data<float>();
        for (int64_t i = 0; i < numPoints; i++) op[i] = logit01;
    }

    // 4D trajectory coefficients, all zero: mu(t) == mu and q(t) == q at every
    // t, so training starts from exactly the static model and only departs from
    // it as the coefficients pick up gradient.
    if (is4D()) {
        deform = gpu_zeros({numPoints, (int64_t)deformStride()}, DType::Float32);
        means_t = gpu_empty({numPoints, 3}, DType::Float32);
        if (hasRot()) quats_t = gpu_empty({numPoints, 4}, DType::Float32);
    }

    // Background color — default is magenta (high-contrast against typical scenes,
    // makes under-reconstructed regions obvious during training)
    backgroundColor = gpu_empty({3}, DType::Float32);
    static const float defaultBg[3] = {0.6130f, 0.0101f, 0.3984f};
    memcpy(backgroundColor.data_ptr(), bgColor ? bgColor : defaultBg, 3 * sizeof(float));
    setupOptimizers();
}

void Model::setupOptimizers(){
    releaseOptimizers();


    num_active = means.size(0);
    buf_capacity = num_active * 4;
    auto allocBuf = [&](MTensor &buf, const MTensor &param) {
        auto shape = param.shape();
        shape[0] = buf_capacity;
        buf = gpu_zeros(shape, DType::Float32);
        memcpy(buf.data_ptr(), param.data_ptr(), param.nbytes());
    };
    allocBuf(means_buf, means);
    allocBuf(scales_buf, scales);
    allocBuf(quats_buf, quats);
    allocBuf(featuresDc_buf, featuresDc);
    allocBuf(featuresRest_buf, featuresRest);
    allocBuf(opacities_buf, opacities);
    if (is4D()) allocBuf(deform_buf, deform);

    static constexpr float lr_init[] = {0.00016f, 0.005f, 0.001f, 0.0025f, 0.000125f, 0.05f, 0.0f};
    MTensor *params[] = {&means, &scales, &quats, &featuresDc, &featuresRest, &opacities, &deform};
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        if (!params[g]->defined()) continue;   // trajectory group on a static run
        auto shape = params[g]->shape();
        shape[0] = buf_capacity;
        adam_exp_avg_buf[g] = gpu_zeros(shape, DType::Float32);
        adam_exp_avg_sq_buf[g] = gpu_zeros(shape, DType::Float32);
        adam_lr[g] = lr_init[g];
    }
    if (is4D()) adam_lr[6] = dcfg.lr;
    adam_step_count = 0;
    means_lr_init = 0.00016f;
    means_lr_final = 0.0000016f;

    densify_split_flag = gpu_zeros({buf_capacity}, DType::Int32);
    densify_dup_flag = gpu_zeros({buf_capacity}, DType::Int32);
    densify_split_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    densify_dup_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    densify_keep_flag = gpu_zeros({buf_capacity}, DType::Int32);
    densify_keep_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    int max_blocks = (buf_capacity + 1023) / 1024;
    densify_block_totals = gpu_zeros({max_blocks}, DType::Int32);
    // Scratch has to fit the widest per-gaussian buffer; with 4D enabled the
    // trajectory stride (54 by default) overtakes featuresRest (45).
    int64_t fr_stride = featuresRest.numel() / featuresRest.size(0);
    int64_t max_stride = std::max(fr_stride, (int64_t)deformStride());
    densify_compact_scratch = gpu_zeros({(int64_t)buf_capacity * max_stride}, DType::Float32);
    densify_random_samples = gpu_zeros({buf_capacity, 3}, DType::Float32);

    refreshViews();
    rebuildNeighbors();
}

void Model::releaseOptimizers(){
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        adam_exp_avg[g].reset(); adam_exp_avg_sq[g].reset();
        adam_exp_avg_buf[g].reset(); adam_exp_avg_sq_buf[g].reset();
    }
    means_buf.reset(); scales_buf.reset(); quats_buf.reset();
    featuresDc_buf.reset(); featuresRest_buf.reset(); opacities_buf.reset();
    deform_buf.reset(); means_t.reset(); quats_t.reset();
    velocity.reset(); v_vel.reset(); v_mu_extra.reset();
    flow2d.reset(); v_flow2d.reset(); neighbors.reset();
    temporal_w.reset(); v_temporal_w.reset();
    densify_split_flag.reset(); densify_dup_flag.reset();
    densify_split_prefix.reset(); densify_dup_prefix.reset();
    densify_keep_flag.reset(); densify_keep_prefix.reset();
    densify_block_totals.reset(); densify_compact_scratch.reset(); densify_random_samples.reset();
}

void Model::schedulersStep(int step){
    float t = std::clamp((float)step / (float)maxSteps, 0.f, 1.f);
    adam_lr[0] = std::exp(std::log(means_lr_init) * (1.f - t) + std::log(means_lr_final) * t);
}

void Model::refreshViews(){
    means = means_buf.view(num_active);
    scales = scales_buf.view(num_active);
    quats = quats_buf.view(num_active);
    featuresDc = featuresDc_buf.view(num_active);
    featuresRest = featuresRest_buf.view(num_active);
    opacities = opacities_buf.view(num_active);
    if (is4D()) {
        deform = deform_buf.view(num_active);
        // These are per-step scratch, not densified buffers. refreshViews only
        // runs on densification, so reallocating here is not on the hot path.
        means_t = gpu_empty({(int64_t)num_active, 3}, DType::Float32);
        if (hasRot()) quats_t = gpu_empty({(int64_t)num_active, 4}, DType::Float32);
        if (dcfg.anyPhase3()) {
            velocity = gpu_zeros({(int64_t)num_active, 3}, DType::Float32);
            v_vel = gpu_zeros({(int64_t)num_active, 3}, DType::Float32);
            v_mu_extra = gpu_zeros({(int64_t)num_active, 3}, DType::Float32);
        }
        if (dcfg.flow) {
            flow2d = gpu_zeros({(int64_t)num_active, 2}, DType::Float32);
            v_flow2d = gpu_zeros({(int64_t)num_active, 2}, DType::Float32);
        }
        if (dcfg.temporal()) {
            temporal_w = gpu_zeros({(int64_t)num_active, 1}, DType::Float32);
            v_temporal_w = gpu_zeros({(int64_t)num_active, 1}, DType::Float32);
        }
    }
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        if (!adam_exp_avg_buf[g].defined()) continue;
        adam_exp_avg[g] = adam_exp_avg_buf[g].view(num_active);
        adam_exp_avg_sq[g] = adam_exp_avg_sq_buf[g].view(num_active);
    }
}

void Model::ensureCapacity(int needed){
    if (needed <= buf_capacity) return;
    int new_cap = std::max(needed, buf_capacity * 2);

    auto grow = [&](MTensor &buf) {
        auto shape = buf.shape();
        shape[0] = new_cap;
        MTensor new_buf = gpu_zeros(shape, DType::Float32);
        size_t copy_bytes = num_active * buf.stride0() * sizeof(float);
        memcpy(new_buf.data_ptr(), buf.data_ptr(), copy_bytes);
        buf = new_buf;
    };
    grow(means_buf); grow(scales_buf); grow(quats_buf);
    grow(featuresDc_buf); grow(featuresRest_buf); grow(opacities_buf);
    if (is4D()) grow(deform_buf);
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        if (!adam_exp_avg_buf[g].defined()) continue;
        grow(adam_exp_avg_buf[g]);
        grow(adam_exp_avg_sq_buf[g]);
    }
    densify_split_flag = gpu_zeros({new_cap}, DType::Int32);
    densify_dup_flag = gpu_zeros({new_cap}, DType::Int32);
    densify_split_prefix = gpu_zeros({new_cap}, DType::Int32);
    densify_dup_prefix = gpu_zeros({new_cap}, DType::Int32);
    densify_keep_flag = gpu_zeros({new_cap}, DType::Int32);
    densify_keep_prefix = gpu_zeros({new_cap}, DType::Int32);
    int max_blocks = (new_cap + 1023) / 1024;
    densify_block_totals = gpu_zeros({max_blocks}, DType::Int32);
    int64_t fr_stride = featuresRest_buf.stride0();
    int64_t max_stride = std::max(fr_stride, (int64_t)deformStride());
    densify_compact_scratch = gpu_zeros({(int64_t)new_cap * max_stride}, DType::Float32);
    densify_random_samples = gpu_zeros({new_cap, 3}, DType::Float32);

    buf_capacity = new_cap;
    refreshViews();
}

int Model::getDownscaleFactor(int step) {
    int remaining = numDownscales - step / resolutionSchedule;
    return 1 << std::max(remaining, 0);
}

void Model::afterTrain(int step){
    if (!radii.defined()) return;

    if (step % refineEvery == 0 && step > warmupLength){
        int resetInterval = resetAlphaEvery * refineEvery;
        bool doDensification = step < stopSplitAt && step % resetInterval > numCameras + refineEvery;

        if (doDensification){
            int numPointsBefore = num_active;
            ensureCapacity(3 * num_active);  // worst case: every gaussian splits

            // Fill random samples for splits (CPU randn, shared memory)
            {
                std::mt19937 rng(step);
                std::normal_distribution<float> dist(0.0f, 1.0f);
                float *p = densify_random_samples.data<float>();
                for (int64_t i = 0; i < 2 * num_active * 3; i++) p[i] = dist(rng);
            }

            float half_max_dim = 0.5f * static_cast<float>((std::max)(lastWidth, lastHeight));
            int check_screen = (step < stopScreenSizeAt) ? 1 : 0;
            // Huge-gaussian cull, threshold scaled by scene extent as in
            // reference 3DGS (0.1 * cameras_extent). The previous absolute
            // 0.5-unit threshold deleted the legitimate large ground-plane
            // gaussians of unbounded scenes; gating stays at the first
            // opacity reset like splatfacto.
            bool checkHuge = step > refineEvery * resetAlphaEvery;
            float cullScaleThresh = 0.1f * sceneExtent;
            int fr_stride = (int)featuresRest_buf.stride0();

            int new_count = msplat_densify(
                num_active, buf_capacity,
                densifyGradThresh, densifySizeThresh, splitScreenSize, check_screen,
                0.1f, cullScaleThresh, 0.15f, checkHuge ? 1 : 0,
                xysGradNorm, visCounts, max2DSize, half_max_dim,
                means_buf, scales_buf, quats_buf,
                featuresDc_buf, featuresRest_buf, opacities_buf, fr_stride,
                adam_exp_avg_buf, adam_exp_avg_sq_buf,
                densify_split_flag, densify_dup_flag,
                densify_split_prefix, densify_dup_prefix,
                densify_keep_flag, densify_keep_prefix,
                densify_block_totals, densify_compact_scratch,
                densify_random_samples,
                deform_buf, is4D() ? deformStride() : 0
            );

            num_active = new_count;
            refreshViews();
            rebuildNeighbors();   // the kNN graph is stale once the point set changed
            std::cout << "Densified: " << numPointsBefore << " -> " << num_active << " gaussians" << std::endl;
        }

        if (step < stopSplitAt && step % resetInterval == refineEvery){
            msplat_gpu_sync();
            constexpr float resetLogit = -1.3862943611198906f;
            float *op = opacities.data<float>();
            for (int64_t i = 0; i < opacities.numel(); i++)
                if (op[i] > resetLogit) op[i] = resetLogit;

            adam_exp_avg[5].zero();
            adam_exp_avg_sq[5].zero();
            fprintf(stderr, "Opacity reset at step %d\n", step);
        }

        xysGradNorm.reset();
        visCounts.reset();
        max2DSize.reset();
    }
}

void Model::save(const std::string &filename, int step) {
    std::string ext = fs::path(filename).extension().string();
    if (ext == ".splat")
        saveSplat(filename);
    else
        savePly(filename, step);
    fprintf(stderr, "Saved %s\n", filename.c_str());
}

void Model::savePly(const std::string &filename, int step){
    GaussianParams p{means, scales, quats, featuresDc, featuresRest, opacities,
                     scale, {translation[0], translation[1], translation[2]}, keepCrs};
    saveGaussianPly(filename, p, step);
}

void Model::savePlyAt(const std::string &filename, int step, float time, int maxShBases){
    // deformedMeans() returns the canonical means for a static model, so this
    // stays correct (and time-independent) when no trajectory is trained.
    MTensor &m = deformedMeans(time);
    GaussianParams p{m, scales, deformedQuats(), featuresDc, featuresRest, opacities,
                     scale, {translation[0], translation[1], translation[2]}, keepCrs};
    saveGaussianPly(filename, p, step, maxShBases);
}

void Model::savePlySequence(const std::string &dir, const std::string &prefix, int step,
                            int numFrames, float t0, float t1, int maxShBases){
    if (numFrames < 1)
        throw std::runtime_error("savePlySequence: numFrames must be >= 1");
    fs::create_directories(dir);

    for (int f = 0; f < numFrames; f++) {
        // Inclusive of both endpoints, so frame 0 is t0 and the last is t1.
        float t = numFrames == 1 ? t0
                                 : t0 + (t1 - t0) * (float)f / (float)(numFrames - 1);
        std::string idx = std::to_string(f);
        idx.insert(0, idx.size() < 4 ? 4 - idx.size() : 0, '0');
        savePlyAt((fs::path(dir) / (prefix + "_" + idx + ".ply")).string(),
                  step, t, maxShBases);
    }
    fprintf(stderr, "Saved %d frames to %s (t=%.3f..%.3f%s)\n",
            numFrames, dir.c_str(), t0, t1,
            maxShBases == 0 ? ", DC only" : "");
}

void Model::saveSplat(const std::string &filename){
    GaussianParams p{means, scales, quats, featuresDc, featuresRest, opacities,
                     scale, {translation[0], translation[1], translation[2]}, keepCrs};
    saveGaussianSplat(filename, p);
}

int Model::loadPly(const std::string &filename){
    auto g = loadGaussianPly(filename, scale, translation, keepCrs);
    means = g.means;
    scales = g.scales;
    quats = g.quats;
    featuresDc = g.featuresDc;
    featuresRest = g.featuresRest;
    opacities = g.opacities;
    setupOptimizers();
    return g.step;
}

// ── Checkpoint save/load ────────────────────────────────────────────────────

static constexpr uint32_t CKPT_MAGIC = 0x4C50534D; // "MSPL"
// v2: the 4D block stores four trajectory orders (means + rotation) instead of
// two, and `deform` holds the combined means/rotation coefficients.
// v3: a fifth order word carries the Phase-4 temporal-envelope flag, which
// changes the stride of `deform` and so cannot be inferred from the file.
static constexpr uint32_t CKPT_VERSION = 3;

static void writeTensor(std::ofstream &f, MTensor &t) {
    uint32_t ndim = t.ndim();
    f.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    for (int i = 0; i < (int)ndim; i++) {
        int64_t s = t.size(i);
        f.write(reinterpret_cast<const char*>(&s), sizeof(s));
    }
    uint64_t bytes = t.nbytes();
    f.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
    f.write(reinterpret_cast<const char*>(t.data_ptr()), bytes);
}

static MTensor readTensor(std::ifstream &f) {
    uint32_t ndim;
    f.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    std::vector<int64_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; i++)
        f.read(reinterpret_cast<char*>(&shape[i]), sizeof(int64_t));
    uint64_t bytes;
    f.read(reinterpret_cast<char*>(&bytes), sizeof(bytes));
    MTensor t = gpu_empty(shape, DType::Float32);
    f.read(reinterpret_cast<char*>(t.data_ptr()), bytes);
    return t;
}

void Model::saveCheckpoint(const std::string &filename, int step) {
    msplat_gpu_sync();

    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open checkpoint file for writing: " + filename);

    // Header
    f.write(reinterpret_cast<const char*>(&CKPT_MAGIC), sizeof(CKPT_MAGIC));
    f.write(reinterpret_cast<const char*>(&CKPT_VERSION), sizeof(CKPT_VERSION));

    // Scalar state
    uint32_t u;
    u = (uint32_t)step;            f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    u = (uint32_t)num_active;      f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    u = (uint32_t)shDegree;        f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    u = (uint32_t)adam_step_count;  f.write(reinterpret_cast<const char*>(&u), sizeof(u));

    // Adam learning rates
    f.write(reinterpret_cast<const char*>(adam_lr), sizeof(adam_lr));
    f.write(reinterpret_cast<const char*>(&means_lr_init), sizeof(means_lr_init));
    f.write(reinterpret_cast<const char*>(&means_lr_final), sizeof(means_lr_final));

    // Gaussian parameters (views — only num_active elements)
    writeTensor(f, means);
    writeTensor(f, scales);
    writeTensor(f, quats);
    writeTensor(f, featuresDc);
    writeTensor(f, featuresRest);
    writeTensor(f, opacities);

    // Optimizer state for the six static groups (unchanged on-disk layout)
    for (int g = 0; g < 6; g++) writeTensor(f, adam_exp_avg[g]);
    for (int g = 0; g < 6; g++) writeTensor(f, adam_exp_avg_sq[g]);

    // 4D trajectory state, appended behind a flag so static checkpoints keep
    // exactly the format they had before Phase 2.
    uint32_t has4D = is4D() ? 1u : 0u;
    f.write(reinterpret_cast<const char*>(&has4D), sizeof(has4D));
    if (has4D) {
        uint32_t ord[5] = {(uint32_t)dcfg.ord.nPoly, (uint32_t)dcfg.ord.nFourier,
                           (uint32_t)dcfg.ord.qPoly, (uint32_t)dcfg.ord.qFourier,
                           dcfg.ord.temporal ? 1u : 0u};
        f.write(reinterpret_cast<const char*>(ord), sizeof(ord));
        writeTensor(f, deform);
        writeTensor(f, adam_exp_avg[6]);
        writeTensor(f, adam_exp_avg_sq[6]);
    }

    f.close();
    std::cout << "Checkpoint saved: " << filename << " (step " << step
              << ", " << num_active << " gaussians, "
              << fs::file_size(filename) / (1024*1024) << " MB)" << std::endl;
}

int Model::loadCheckpoint(const std::string &filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open checkpoint file: " + filename);

    // Header
    uint32_t magic, version;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (magic != CKPT_MAGIC) throw std::runtime_error("Not a valid msplat checkpoint file");
    if (version != CKPT_VERSION) throw std::runtime_error("Unsupported checkpoint version: " + std::to_string(version));

    // Scalar state
    uint32_t step, numPts, shDeg, adamSteps;
    f.read(reinterpret_cast<char*>(&step), sizeof(step));
    f.read(reinterpret_cast<char*>(&numPts), sizeof(numPts));
    f.read(reinterpret_cast<char*>(&shDeg), sizeof(shDeg));
    f.read(reinterpret_cast<char*>(&adamSteps), sizeof(adamSteps));

    f.read(reinterpret_cast<char*>(adam_lr), sizeof(adam_lr));
    f.read(reinterpret_cast<char*>(&means_lr_init), sizeof(means_lr_init));
    f.read(reinterpret_cast<char*>(&means_lr_final), sizeof(means_lr_final));
    adam_step_count = (int)adamSteps;

    // Gaussian parameters — read into fresh tensors
    means = readTensor(f);
    scales = readTensor(f);
    quats = readTensor(f);
    featuresDc = readTensor(f);
    featuresRest = readTensor(f);
    opacities = readTensor(f);

    // Optimizer state for the six static groups
    for (int g = 0; g < 6; g++) adam_exp_avg[g] = readTensor(f);
    for (int g = 0; g < 6; g++) adam_exp_avg_sq[g] = readTensor(f);

    // 4D trajectory state (see saveCheckpoint). Absent in pre-Phase-2 files,
    // which simply end here.
    uint32_t has4D = 0;
    f.read(reinterpret_cast<char*>(&has4D), sizeof(has4D));
    if (f && has4D) {
        uint32_t ord[5] = {};
        f.read(reinterpret_cast<char*>(ord), sizeof(ord));
        dcfg.ord.nPoly = (int)ord[0];
        dcfg.ord.nFourier = (int)ord[1];
        dcfg.ord.qPoly = (int)ord[2];
        dcfg.ord.qFourier = (int)ord[3];
        dcfg.ord.temporal = ord[4] != 0;
        deform = readTensor(f);
        adam_exp_avg[6] = readTensor(f);
        adam_exp_avg_sq[6] = readTensor(f);
    } else {
        dcfg.ord = DeformOrders{};
    }

    f.close();

    // Rebuild backing buffers with loaded data (don't call setupOptimizers —
    // it would zero the optimizer state we just loaded)
    num_active = (int)numPts;
    buf_capacity = num_active * 4;

    // Copy gaussian params into oversized backing buffers
    auto allocBuf = [&](MTensor &buf, const MTensor &param) {
        auto shape = param.shape();
        shape[0] = buf_capacity;
        buf = gpu_zeros(shape, DType::Float32);
        memcpy(buf.data_ptr(), param.data_ptr(), param.nbytes());
    };
    allocBuf(means_buf, means);
    allocBuf(scales_buf, scales);
    allocBuf(quats_buf, quats);
    allocBuf(featuresDc_buf, featuresDc);
    allocBuf(featuresRest_buf, featuresRest);
    allocBuf(opacities_buf, opacities);
    if (is4D()) allocBuf(deform_buf, deform);

    // Copy optimizer state into oversized backing buffers
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        if (!adam_exp_avg[g].defined()) continue;
        auto shape = adam_exp_avg[g].shape();
        shape[0] = buf_capacity;
        MTensor avg_buf = gpu_zeros(shape, DType::Float32);
        MTensor sq_buf = gpu_zeros(shape, DType::Float32);
        memcpy(avg_buf.data_ptr(), adam_exp_avg[g].data_ptr(), adam_exp_avg[g].nbytes());
        memcpy(sq_buf.data_ptr(), adam_exp_avg_sq[g].data_ptr(), adam_exp_avg_sq[g].nbytes());
        adam_exp_avg_buf[g] = avg_buf;
        adam_exp_avg_sq_buf[g] = sq_buf;
    }

    // Allocate densification scratch buffers
    densify_split_flag = gpu_zeros({buf_capacity}, DType::Int32);
    densify_dup_flag = gpu_zeros({buf_capacity}, DType::Int32);
    densify_split_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    densify_dup_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    densify_keep_flag = gpu_zeros({buf_capacity}, DType::Int32);
    densify_keep_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    int max_blocks = (buf_capacity + 1023) / 1024;
    densify_block_totals = gpu_zeros({max_blocks}, DType::Int32);
    int64_t fr_stride = featuresRest.numel() / featuresRest.size(0);
    densify_compact_scratch = gpu_zeros({(int64_t)buf_capacity * fr_stride}, DType::Float32);
    densify_random_samples = gpu_zeros({buf_capacity, 3}, DType::Float32);

    refreshViews();

    std::cout << "Checkpoint loaded: " << filename << " (step " << step
              << ", " << num_active << " gaussians)" << std::endl;

    return (int)step;
}

Model::CamSetup Model::prepareCam(Camera& cam, int step) {
    const float sf = getDownscaleFactor(step);
    CamSetup s;
    s.fx = cam.fx / sf; s.fy = cam.fy / sf;
    s.cx = cam.cx / sf; s.cy = cam.cy / sf;
    s.height = static_cast<int>(cam.height / sf);
    s.width = static_cast<int>(cam.width / sf);

    float fovX = 2.0f * std::atan(s.width / (2.0f * s.fx));
    float fovY = 2.0f * std::atan(s.height / (2.0f * s.fy));

    if (!cam.cachedViewMat.defined() || cam.cachedFovX != fovX || cam.cachedFovY != fovY) {
        const float *d = cam.camToWorld;
        float R[3][3], Rinv[3][3], T[3], Tinv[3];
        for (int i = 0; i < 3; i++) {
            R[i][0] = d[i*4+0]; R[i][1] = -d[i*4+1]; R[i][2] = -d[i*4+2]; T[i] = d[i*4+3];
        }
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Rinv[i][j] = R[j][i];
        for (int i = 0; i < 3; i++) Tinv[i] = -(Rinv[i][0]*T[0] + Rinv[i][1]*T[1] + Rinv[i][2]*T[2]);
        float vm[16] = { Rinv[0][0],Rinv[0][1],Rinv[0][2],Tinv[0], Rinv[1][0],Rinv[1][1],Rinv[1][2],Tinv[1], Rinv[2][0],Rinv[2][1],Rinv[2][2],Tinv[2], 0,0,0,1 };
        float t_p = 0.001f * std::tan(0.5f * fovY), r_p = 0.001f * std::tan(0.5f * fovX);
        float pm[16] = { 0.001f/r_p,0,0,0, 0,0.001f/t_p,0,0, 0,0,(1000.0f+0.001f)/(1000.0f-0.001f),-1000.0f*0.001f/(1000.0f-0.001f), 0,0,1,0 };
        float pvm[16] = {};
        for (int i=0;i<4;i++) for (int j=0;j<4;j++) for (int k=0;k<4;k++) pvm[i*4+j] += pm[i*4+k] * vm[k*4+j];

        cam.cachedViewMat = gpu_empty({4, 4}, DType::Float32);
        memcpy(cam.cachedViewMat.data_ptr(), vm, sizeof(vm));
        cam.cachedProjViewMat = gpu_empty({4, 4}, DType::Float32);
        memcpy(cam.cachedProjViewMat.data_ptr(), pvm, sizeof(pvm));
        cam.cachedCamPos[0] = T[0]; cam.cachedCamPos[1] = T[1]; cam.cachedCamPos[2] = T[2];
        cam.cachedFovX = fovX; cam.cachedFovY = fovY;
    }

    s.degreesToUse = (std::min<int>)(step / shDegreeInterval, shDegree);
    int b = featuresRest.size(-2) + 1;
    s.degree = (b <= 1) ? 0 : (b <= 4) ? 1 : (b <= 9) ? 2 : (b <= 16) ? 3 : 4;
    s.tileBounds = std::make_tuple(
        (s.width + BLOCK_X - 1) / BLOCK_X,
        (s.height + BLOCK_Y - 1) / BLOCK_Y, 1);
    s.cam_pos[0] = cam.cachedCamPos[0];
    s.cam_pos[1] = cam.cachedCamPos[1];
    s.cam_pos[2] = cam.cachedCamPos[2];

    return s;
}

// Time is normalised to [0,1]; tau recentres it on the middle of the sequence
// so the polynomial terms stay small and symmetric at both ends.
static inline float deformTau(float time) { return time - 0.5f; }

MTensor& Model::deformedMeans(float time) {
    if (!is4D()) return means;
    msplat_deform_forward(means.size(0), means, quats, deform,
                          deformTau(time), dcfg.ord, means_t, quats_t, temporal_w);
    return means_t;
}

// k nearest neighbours of every gaussian in the CANONICAL means, for L_rigid.
// Rebuilt after every densification rather than every step: a kNN query over
// hundreds of thousands of points costs far more than a training step, and
// "locally rigid" is a statement about the canonical neighbourhood anyway.
void Model::rebuildNeighbors() {
    if (!dcfg.rigid) return;
    msplat_gpu_sync();
    int64_t n = means.size(0);
    int k = std::min<int>(dcfg.rigidK, (int)std::max<int64_t>(1, n - 1));
    neighbors = gpu_empty({n, (int64_t)k}, DType::Int32);
    int32_t *out = neighbors.data<int32_t>();

    PointsTensor pt(means.data<float>(), n);
    PointsTensor::KdTree index(3, pt, {10});
    // k+1 because the query point is its own nearest neighbour.
    std::vector<size_t> idx(k + 1);
    std::vector<float> dist(k + 1);
    for (int64_t i = 0; i < n; i++) {
        size_t found = index.knnSearch(&means.data<float>()[i * 3], k + 1,
                                       idx.data(), dist.data());
        int w = 0;
        for (size_t j = 0; j < found && w < k; j++) {
            if ((int64_t)idx[j] == i) continue;
            out[i * k + w++] = (int32_t)idx[j];
        }
        while (w < k) out[i * k + w++] = -1;   // fewer than k points available
    }
}

MTensor Model::render(Camera& cam, int step){
    auto s = prepareCam(cam, step);
    MTensor &m = deformedMeans(cam.time);
    return msplat_render(
        means.size(0), m, scales, 1.0f,
        deformedQuats(), cam.cachedViewMat, cam.cachedProjViewMat, s.fx, s.fy, s.cx, s.cy,
        s.height, s.width, s.tileBounds, 0.01f,
        s.degree, s.degreesToUse, s.cam_pos, featuresDc, featuresRest,
        opacities, backgroundColor);
}

MTensor Model::renderFlow(Camera& cam, Camera& nextCam, int step, float dt){
    if (!is4D())
        throw std::runtime_error("renderFlow: model has no trajectory (not a 4D run)");
    auto s = prepareCam(cam, step);
    prepareCam(nextCam, step);
    // The colour pass is what fills the sorted tile lists the flow compositing
    // walks; without it the flow render would read stale or empty bins.
    (void)render(cam, step);

    int n = means.size(0);
    if (!velocity.defined() || velocity.size(0) != n)
        velocity = gpu_zeros({(int64_t)n, 3}, DType::Float32);
    if (!flow2d.defined() || flow2d.size(0) != n)
        flow2d = gpu_zeros({(int64_t)n, 2}, DType::Float32);
    MTensor none;
    msplat_velocity_field(n, deform, deformTau(cam.time), dcfg.ord, velocity, none, none);
    return msplat_flow_render(n, means_t, velocity, msplat_forward_radii(),
                              cam.cachedProjViewMat, nextCam.cachedProjViewMat,
                              s.height, s.width, s.cx, s.cy, dt, flow2d);
}

void Model::fullIteration(Camera& cam, int step, MTensor &gt, float ssimWeight,
                          Camera* nextCam){
    auto s = prepareCam(cam, step);
    lastHeight = s.height; lastWidth = s.width;
    int numPoints = means.size(0);

    // Initialize SSIM window (once)
    if (!window2d.defined()) {
        auto w = createSSIMWindow(11, 1.5f);
        window2d = gpu_empty({11, 11}, DType::Float32);
        memcpy(window2d.data_ptr(), w.data(), w.size() * sizeof(float));
    }

    adam_step_count++;
    float bc1 = 1.0f - std::pow(adam_beta1, adam_step_count);
    float bc2 = 1.0f - std::pow(adam_beta2, adam_step_count);
    // Only the six static groups go through the fused train step; the
    // trajectory group is stepped separately after the backward, because its
    // gradient is a chain rule on v_mean3d rather than a buffer the fused
    // kernel produces.
    constexpr int N_FUSED_GROUPS = 6;
    MTensor adam_p[N_FUSED_GROUPS];
    MTensor adam_ea[N_FUSED_GROUPS], adam_eas[N_FUSED_GROUPS];
    float adam_ss[N_FUSED_GROUPS], adam_bc2s[N_FUSED_GROUPS];
    MTensor *params[] = {&means, &scales, &quats, &featuresDc, &featuresRest, &opacities};
    for (int i = 0; i < N_FUSED_GROUPS; ++i) {
        adam_p[i] = *params[i];
        adam_ea[i] = adam_exp_avg[i];
        adam_eas[i] = adam_exp_avg_sq[i];
        adam_ss[i] = adam_lr[i] / bc1;
        adam_bc2s[i] = std::sqrt(bc2);
    }

    if (!xysGradNorm.defined()) {
    
        xysGradNorm = gpu_zeros({numPoints}, DType::Float32);
        visCounts = gpu_zeros({numPoints}, DType::Float32);
        max2DSize = gpu_zeros({numPoints}, DType::Float32);
    }

    float invMaxDim = 1.0f / static_cast<float>((std::max)(lastHeight, lastWidth));
    float lossInvN = 1.0f / (float)(s.height * s.width * 3);

    // Feed mu(t) to the projection while Adam still updates the canonical means:
    // means3d and adam_params[0] are separate arguments, and d mu(t)/d mu = I
    // makes the gradient the fused step computes correct for both.
    MTensor &m = deformedMeans(cam.time);

    // Depth supervision and opacity entropy run inside the fused step (the
    // depth gradient has to reach v_depth before the projection backward turns
    // it into a gradient on the means), so they are published rather than
    // passed. Both are set on EVERY iteration, including to "off" — a target
    // left standing would supervise the next camera with this camera's depth.
    //
    // The width guard is the same one the flow pass uses: ground-truth depth is
    // on the full-resolution pixel grid, and while progressive downscaling is
    // active there is nothing to compare against.
    MTensor *gtDepth = nullptr;
    if (dcfg.depth && s.width == cam.width && s.height == cam.height) {
        gtDepth = cam.getGPUDepth();
        if (gtDepth) depthSteps++; else depthSkippedSteps++;
    } else if (dcfg.depth) {
        depthSkippedSteps++;
    }
    msplat_set_depth_target(gtDepth, dcfg.depthWeight, dcfg.depthMinCoverage);
    msplat_set_opacity_entropy(dcfg.opacityEntropyWeight);

    auto [r, loss] = msplat_train_step(
        numPoints, m, scales, 1.0f,
        deformedQuats(), cam.cachedViewMat, cam.cachedProjViewMat, s.fx, s.fy, s.cx, s.cy,
        s.height, s.width, s.tileBounds, 0.01f,
        s.degree, s.degreesToUse, s.cam_pos, featuresDc, featuresRest,
        opacities, backgroundColor, gt, window2d, ssimWeight,
        lossInvN, (int)featuresRest.size(-2),
        N_FUSED_GROUPS,
        adam_p, adam_ea, adam_eas,
        adam_ss, adam_bc2s,
        adam_beta1, adam_beta2, adam_eps,
        visCounts, xysGradNorm, max2DSize, invMaxDim);

    if (gtDepth || dcfg.opacityEntropyWeight > 0.0f)
        msplat_last_depth_losses(&lastDepthLoss, &lastOpacityEntropy);

    if (is4D()) {
        // ── Phase 3 ──────────────────────────────────────────────────────────
        // Everything below only accumulates into v_vel / v_mu_extra; the single
        // Adam step for the whole coefficient block comes afterwards. Two Adam
        // steps on the same tensor in one iteration would corrupt the moments.
        MTensor noTensor;
        bool ranPhase3 = false;
        if (dcfg.anyPhase3()) {
            msplat_velocity_field(numPoints, deform, deformTau(cam.time), dcfg.ord,
                                  velocity, v_vel, v_mu_extra);
            ranPhase3 = true;
        }

        if (dcfg.flow && nextCam && cam.hasFlow()) {
            MTensor *gtFlow = cam.getGPUFlow();
            // The progressive downscale changes the pixel grid the flow lives on;
            // rather than resample the ground truth, skip those steps. With
            // --num-downscales 0 (the recipe for every 4D run here) this never
            // triggers after the first resolution step.
            if (gtFlow && s.width == cam.width && s.height == cam.height) {
                prepareCam(*nextCam, step);   // makes sure its proj matrix is built
                lastFlowLoss = msplat_flow_step(
                    numPoints, m, velocity, r,
                    cam.cachedProjViewMat, nextCam->cachedProjViewMat,
                    s.height, s.width, s.cx, s.cy, cam.flowDt,
                    *gtFlow, dcfg.flowWeight, dcfg.flowMinCoverage,
                    v_vel, v_mu_extra, flow2d, v_flow2d);
                flowSteps++;
            } else {
                flowSkippedSteps++;
            }
        }

        if (dcfg.rigid && neighbors.defined() && neighbors.size(0) == numPoints) {
            int k = (int)neighbors.size(1);
            lastRigidLoss = msplat_rigid_step(
                numPoints, means, velocity, neighbors, k, dcfg.rigidBeta,
                dcfg.rigidWeight / (float)(k * numPoints), v_vel);
        }

        // Phase 4: the envelope's gradient has to be extracted from the opacity
        // gradient BEFORE the deform backward consumes it, and after the train
        // step produced it.
        if (dcfg.temporal() && temporal_w.defined()) {
            msplat_temporal_opacity_fixup(numPoints, opacities, temporal_w,
                                          msplat_train_v_opacity(), v_temporal_w);
            // Reporting only, on the cadence the CLI prints at: how selective the
            // envelope has become. offFrac is the paper's efficiency claim —
            // the share of gaussians this timestamp does not have to composite.
            if (step % 100 == 0) {
                msplat_gpu_sync();
                const float *w = temporal_w.data<float>();
                double sum = 0.0;
                long off = 0;
                for (int i = 0; i < numPoints; i++) {
                    sum += w[i];
                    if (w[i] < 0.01f) off++;
                }
                lastTemporalMean = (float)(sum / std::max(numPoints, 1));
                lastTemporalOffFrac = (float)off / (float)std::max(numPoints, 1);
            }
        }

        msplat_deform_backward_adam(
            numPoints, msplat_train_v_mean3d(), msplat_train_v_quat(), deform,
            adam_exp_avg[6], adam_exp_avg_sq[6],
            deformTau(cam.time), dcfg.ord,
            adam_lr[6] / bc1, dcfg.rotLr / bc1,
            adam_beta1, adam_beta2, std::sqrt(bc2), adam_eps,
            ranPhase3 ? v_mu_extra : noTensor,
            ranPhase3 ? v_vel : noTensor,
            dcfg.temporal() ? v_temporal_w : noTensor, dcfg.tempLr / bc1,
            dcfg.l1);
    }

    radii = r;

    // MSPLAT_DEBUG_MEANS=1 syncs and scans the means every step — expensive, but
    // it is what localised the SH NaN: a healthy run keeps max|means| at the
    // scene extent, a broken one collapses to +-lr or goes NaN on the first step.
    if (std::getenv("MSPLAT_DEBUG_MEANS")) {
        msplat_gpu_sync();
        const float *a = means.data<float>();
        double mx = 0; int64_t n = means.numel(); int64_t nbad = 0;
        for (int64_t i = 0; i < n; i++) {
            mx = std::max(mx, (double)std::abs(a[i]));
            if (!std::isfinite(a[i])) nbad++;
        }
        fprintf(stderr, "step=%d 4d=%d N=%lld means[0..2]=%g,%g,%g max|means|=%g nonfinite=%lld\n",
                step, (int)is4D(), (long long)means.size(0), a[0], a[1], a[2], mx, (long long)nbad);
    }
}
