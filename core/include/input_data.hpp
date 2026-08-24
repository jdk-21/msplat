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
    // Physical camera this frame came from, for multi-camera rigs where the same
    // camera contributes many timestamps. -1 = the loader does not know, and
    // consumers fall back to matching extrinsics. Phase 3 needs it to pair flow
    // within a camera instead of across the rig.
    int camId = -1;
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

    // ── Depth supervision (Phase 4) ─────────────────────────────────────────
    // Ground-truth depth for THIS frame, in the dataset's own world units.
    // depthScale converts them to the scaled world the model is trained in —
    // autoScaleAndCenter divides every camera position by max|coordinate|, and
    // a depth that missed that factor would pull the geometry to the wrong
    // distance while looking perfectly well-formed.
    std::string depthPath;
    float depthScale = 1.0f;
    MTensor gpuDepth;  // (H, W): metric depth, 0 = no ground truth
    bool hasDepth() const { return !depthPath.empty(); }
    MTensor* getGPUDepth();

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

// Phase 4: point every camera at its ground-truth depth file under
// <datasetPath>/depth/<image stem>.dpt, mirroring the image tree the same way
// the flow lookup does. sceneScale is InputData::scale, the factor
// autoScaleAndCenter applied to the camera positions; it is stored per camera
// and applied when the file is read. Returns the number of linked frames.
//
// Unlike flow this needs no pairing and no next frame, so it is safe to call on
// the test list too — useful for measuring geometry error on a held-out view,
// which is the one number that says whether the reconstruction is a surface or
// a cloud. Training only ever reads it from the train list.
int attachDepthToCameras(std::vector<Camera> &cams, const std::string &datasetPath,
                         float sceneScale);

#endif
