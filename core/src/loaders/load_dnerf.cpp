#include "loaders.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <random>
#include <cmath>

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
static constexpr int64_t INIT_NUM_POINTS = 100000;

static void loadSplit(const std::string &projectRoot, const std::string &file,
                      bool isTest, bool whiteBackground, InputData &data) {
    auto path = fs::path(projectRoot) / file;
    if (!fs::exists(path)) return;

    std::ifstream f(path.string());
    json j = json::parse(f);

    const float angleX = j.value("camera_angle_x", 0.0f);
    if (angleX <= 0.0f)
        throw std::runtime_error("D-NeRF: missing or invalid camera_angle_x in " + file);

    for (auto &frame : j["frames"]) {
        Camera cam;
        cam.width = cam.height = DNERF_NOMINAL_RES;
        // Blender's camera_angle_x is the full horizontal FOV.
        cam.fx = cam.fy = 0.5f * DNERF_NOMINAL_RES / std::tan(0.5f * angleX);
        cam.cx = cam.cy = 0.5f * DNERF_NOMINAL_RES;

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
    loadSplit(projectRoot, "transforms_train.json", false, whiteBackground, data);
    loadSplit(projectRoot, "transforms_test.json", true, whiteBackground, data);
    if (data.cameras.empty())
        throw std::runtime_error("D-NeRF: no frames found in " + projectRoot);
    data.hasExplicitSplit = true;

    // No SfM points — seed a uniform random cloud around the object, the same
    // initialisation the published dynamic-Gaussian baselines use.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pos(-INIT_CUBE_HALF_EXTENT, INIT_CUBE_HALF_EXTENT);
    std::uniform_int_distribution<int> col(0, 255);
    data.points.count = INIT_NUM_POINTS;
    data.points.xyz.resize(INIT_NUM_POINTS * 3);
    data.points.rgb.resize(INIT_NUM_POINTS * 3);
    for (int64_t i = 0; i < INIT_NUM_POINTS * 3; i++) {
        data.points.xyz[i] = pos(rng);
        data.points.rgb[i] = (uint8_t)col(rng);
    }

    autoScaleAndCenter(data);
    return data;
}
