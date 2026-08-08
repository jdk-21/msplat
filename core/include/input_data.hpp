#ifndef INPUT_DATA_H
#define INPUT_DATA_H

#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include "metal_tensor.hpp"

// Simple float32 RGB image — replaces cv::Mat
struct Image {
    std::vector<float> data;  // width * height * 3 floats, RGB, [0,1]
    int width = 0, height = 0;

    bool empty() const { return data.empty(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

struct Camera {
    int width = 0, height = 0;
    float fx = 0, fy = 0, cx = 0, cy = 0;
    float k1 = 0, k2 = 0, k3 = 0, p1 = 0, p2 = 0;
    float camToWorld[16] = {};  // 4x4 row-major, camera-to-world (OpenGL: Y-up, Z-back)
    std::string filePath;

    // ── 4D (Phase 2) ────────────────────────────────────────────────────────
    // Normalized timestamp in [0,1]. Static datasets leave this at 0, which
    // makes every time-dependent term collapse to its canonical value.
    float time = 0.0f;
    // Set by loaders that carry their own train/test split (e.g. D-NeRF)
    // instead of relying on the every-Nth-image rule.
    bool isTest = false;
    // Background to composite RGBA source images over. Synthetic datasets ship
    // transparent PNGs; without this the alpha is dropped and holes read black.
    bool hasBgComposite = false;
    float bgComposite[3] = {};

    // ── Flow splatting (Phase 3) ────────────────────────────────────────────
    // Ground-truth optical flow from THIS frame to the next one in time, in
    // pixels per frame interval. flowNextIdx indexes the train camera list the
    // frontend holds; flowDt is the normalized time between the two frames.
    std::string flowPath;
    int flowNextIdx = -1;
    float flowDt = 0.0f;
    MTensor gpuFlow;   // (H, W, 3): u, v, valid — loaded on first use
    bool hasFlow() const { return flowNextIdx >= 0 && !flowPath.empty(); }
    // Loads flowPath into gpuFlow. Returns nullptr if the file is missing or its
    // resolution does not match the (full-resolution) image.
    MTensor* getGPUFlow();

    Image image;
    std::unordered_map<int, Image> imagePyramids;
    std::unordered_map<int, MTensor> mtensorImageCache;
    MTensor cachedViewMat, cachedProjViewMat;
    float cachedCamPos[3] = {};
    float cachedFovX = 0, cachedFovY = 0;

    void loadImage(float downscaleFactor);
    Image getImage(int downscaleFactor);
    MTensor& getGPUImage(int downscaleFactor);
    bool hasDistortion() const { return k1 != 0 || k2 != 0 || k3 != 0 || p1 != 0 || p2 != 0; }
};

struct Points {
    std::vector<float> xyz;     // N*3 flattened
    std::vector<uint8_t> rgb;   // N*3 flattened
    int64_t count = 0;
};

struct InputData {
    std::vector<Camera> cameras;
    float scale = 1.0f;
    float translation[3] = {};
    Points points;
    // True when the loader already tagged cameras via Camera::isTest.
    bool hasExplicitSplit = false;

    std::tuple<std::vector<Camera>, Camera*> getCameras(bool validate, const std::string &valImage = "random");
    std::tuple<std::vector<Camera>, std::vector<Camera>> splitTrainTest(int testEvery);
    void saveCameras(const std::string &filename, bool keepCrs) const;
};

// Auto-detect format and load dataset
InputData inputDataFromX(const std::string &path, const std::string &colmapImagePath = "",
                         bool whiteBackground = false);

// Phase 3: link consecutive frames and their ground-truth flow files under
// <datasetPath>/flow/<image stem>.flo. Call it on the TRAIN camera list after
// the split — flowNextIdx indexes exactly that vector. Returns the number of
// linked frames.
int attachFlowToCameras(std::vector<Camera> &cams, const std::string &datasetPath);

#endif
