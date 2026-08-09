#include "loaders.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <random>
#include <cmath>
#include <cstdlib>

namespace fs = std::filesystem;
using json = nlohmann::json;

// D-NeRF / Blender-style dynamic scenes.
//
// Layout:
//   <root>/transforms_train.json   frames[] = {file_path, time, transform_matrix}
//   <root>/transforms_test.json
//   <root>/train/r_000.png ...     800x800 RGBA, object on transparent background
//
// Differences from the static Nerfstudio loader:
//   * a per-frame `time` in [0,1] drives the 4D deformation,
//   * intrinsics come from a single global `camera_angle_x`,
//   * train/test split is given by the files, not by the every-Nth rule,
//   * there is no SfM point cloud, so we seed a random one.

// D-NeRF ships 800x800 renders. If the actual file differs, Camera::loadImage
// rescales the intrinsics by the same ratio, so this is only a starting guess.
static constexpr int DNERF_NOMINAL_RES = 800;

// Half-extent of the cube the initial point cloud is drawn from, in D-NeRF world
// units. The synthetic objects all sit inside roughly [-1.3, 1.3]^3 with the
// cameras orbiting at radius ~4.
static constexpr float INIT_CUBE_HALF_EXTENT = 1.3f;

// The published dynamic-Gaussian baselines seed 100k random points here, but
// msplat's rasterizer sorts at most MAX_TILE_ELEMS = 2048 gaussians per 16x16
// tile in threadgroup memory. A dense random cube projects onto very few tiles,
// so 100k points overflow that limit catastrophically on the very first step —
// and the overflow does not merely drop splats, it corrupts the parameter
// buffers (means come back zeroed or NaN). Keep the default under the limit;
// MSPLAT_DNERF_INIT_POINTS raises it for experiments.
static constexpr int64_t INIT_NUM_POINTS = 20000;

static int64_t initNumPoints() {
    if (const char *e = std::getenv("MSPLAT_DNERF_INIT_POINTS")) {
        long long v = atoll(e);
        if (v > 0) return (int64_t)v;
    }
    return INIT_NUM_POINTS;
}

static void loadSplit(const std::string &projectRoot, const std::string &file,
                      bool isTest, bool whiteBackground, InputData &data,
                      float &initExtent, std::string &pointsFile) {
    auto path = fs::path(projectRoot) / file;
    if (!fs::exists(path)) return;

    std::ifstream f(path.string());
    json j = json::parse(f);

    // Optional extensions used by our own multi-camera stage generator and the
    // COLMAP rig converter. D-NeRF itself ships none of them, so the defaults
    // reproduce it exactly.
    const int nomW = j.value("w", DNERF_NOMINAL_RES);
    const int nomH = j.value("h", DNERF_NOMINAL_RES);
    initExtent = j.value("init_extent", initExtent);
    pointsFile = j.value("points_file", pointsFile);

    // A real rig is built from physically distinct cameras, each with its own
    // calibration, so intrinsics may be given per frame. camera_angle_x stays
    // the fallback and is only required when nothing else supplies a focal
    // length — a synthetic dataset where every camera is the same virtual lens.
    const float angleX = j.value("camera_angle_x", 0.0f);
    const bool haveGlobalFocal = angleX > 0.0f;
    const float globalFocal = haveGlobalFocal
                                  ? 0.5f * nomW / std::tan(0.5f * angleX)
                                  : 0.0f;

    for (auto &frame : j["frames"]) {
        Camera cam;
        cam.width = frame.value("w", nomW);
        cam.height = frame.value("h", nomH);

        // Blender's camera_angle_x is the full horizontal FOV; pixels are square,
        // so the vertical focal length is the same number, not one derived from
        // the height.
        const float fx = frame.value("fl_x", globalFocal);
        if (fx <= 0.0f)
            throw std::runtime_error(
                "D-NeRF: " + file + " has neither a valid camera_angle_x nor a "
                "per-frame fl_x");
        cam.fx = fx;
        cam.fy = frame.value("fl_y", fx);
        cam.cx = frame.value("cx", 0.5f * cam.width);
        cam.cy = frame.value("cy", 0.5f * cam.height);

        // Brown-Conrady coefficients, applied by Camera::loadImage, which
        // undistorts and then rewrites the intrinsics to match the crop.
        cam.k1 = frame.value("k1", 0.0f);
        cam.k2 = frame.value("k2", 0.0f);
        cam.k3 = frame.value("k3", 0.0f);
        cam.p1 = frame.value("p1", 0.0f);
        cam.p2 = frame.value("p2", 0.0f);

        cam.camId = frame.value("cam_id", -1);

        // file_path is extension-less ("./train/r_000").
        std::string fp = frame["file_path"].get<std::string>();
        auto full = (fs::path(projectRoot) / fp).string();
        cam.filePath = fs::exists(full) ? full : full + ".png";

        auto &tm = frame["transform_matrix"];
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                cam.camToWorld[r * 4 + c] = tm[r][c].get<float>();

        cam.time = frame.value("time", 0.0f);
        cam.isTest = isTest;
        cam.hasBgComposite = true;
        const float v = whiteBackground ? 1.0f : 0.0f;
        cam.bgComposite[0] = cam.bgComposite[1] = cam.bgComposite[2] = v;

        data.cameras.push_back(cam);
    }
}

InputData loaders::loadDnerf(const std::string &projectRoot, bool whiteBackground) {
    InputData data;
    float initExtent = INIT_CUBE_HALF_EXTENT;
    std::string pointsFile;
    loadSplit(projectRoot, "transforms_train.json", false, whiteBackground, data,
              initExtent, pointsFile);
    loadSplit(projectRoot, "transforms_test.json", true, whiteBackground, data,
              initExtent, pointsFile);
    if (data.cameras.empty())
        throw std::runtime_error("D-NeRF: no frames found in " + projectRoot);
    data.hasExplicitSplit = true;

    // A real capture has been through SfM already, and its sparse cloud is a far
    // better start than a random cube: it sits on actual surfaces, carries
    // measured colour, and — the part that matters on a stage — is shaped like
    // the room instead of like a box centred on the origin. Optional, because a
    // synthetic dataset has no SfM step to take it from.
    if (!pointsFile.empty()) {
        auto p = fs::path(pointsFile);
        if (p.is_relative()) p = fs::path(projectRoot) / p;
        if (!fs::exists(p))
            throw std::runtime_error("D-NeRF: points_file not found: " + p.string());
        data.points = (p.extension() == ".bin") ? readColmapPoints(p.string())
                                                : readPly(p.string());
        if (data.points.count > 0) {
            autoScaleAndCenter(data);
            return data;
        }
        // An empty cloud is a broken conversion, not a reason to silently fall
        // back to noise and let it show up as bad reconstruction much later.
        throw std::runtime_error("D-NeRF: points_file holds no points: " + p.string());
    }

    // No SfM points — seed a uniform random cloud around the object, the same
    // initialisation the published dynamic-Gaussian baselines use.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pos(-initExtent, initExtent);
    std::uniform_int_distribution<int> col(0, 255);
    const int64_t nPts = initNumPoints();
    data.points.count = nPts;
    data.points.xyz.resize(nPts * 3);
    data.points.rgb.resize(nPts * 3);
    for (int64_t i = 0; i < nPts * 3; i++) {
        data.points.xyz[i] = pos(rng);
        data.points.rgb[i] = (uint8_t)col(rng);
    }

    autoScaleAndCenter(data);
    return data;
}
