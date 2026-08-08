#include "input_data.hpp"
#include "loaders.hpp"
#include "msplat.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── Image loading ───────────────────────────────────────────────────────────

void Camera::loadImage(float downscaleFactor) {
    Image raw = imreadRGB(filePath, hasBgComposite ? bgComposite : nullptr);
    if (raw.empty()) return;

    // If actual image dimensions differ from metadata, rescale intrinsics
    if (width > 0 && height > 0 && (raw.width != width || raw.height != height)) {
        float sx = (float)raw.width / (float)width;
        float sy = (float)raw.height / (float)height;
        fx *= sx; fy *= sy; cx *= sx; cy *= sy;
        width = raw.width; height = raw.height;
    } else if (width == 0 || height == 0) {
        width = raw.width; height = raw.height;
    }

    // Downscale
    if (downscaleFactor > 1.0f) {
        int newW = (int)(width / downscaleFactor);
        int newH = (int)(height / downscaleFactor);
        raw = resizeArea(raw, newW, newH);
        float s = 1.0f / downscaleFactor;
        fx *= s; fy *= s; cx *= s; cy *= s;
        width = newW; height = newH;
    }

    // Undistort if needed
    if (hasDistortion()) {
        auto result = undistortImage(raw, fx, fy, cx, cy, k1, k2, p1, p2, k3);
        raw = std::move(result.image);
        fx = result.fx; fy = result.fy;
        cx = result.cx; cy = result.cy;
        width = result.width; height = result.height;
        k1 = k2 = k3 = p1 = p2 = 0;
    }

    image = std::move(raw);
}

Image Camera::getImage(int downscaleFactor) {
    if (downscaleFactor <= 1) return image;

    auto it = imagePyramids.find(downscaleFactor);
    if (it != imagePyramids.end()) return it->second;

    int newW = image.width / downscaleFactor;
    int newH = image.height / downscaleFactor;
    Image scaled = resizeArea(image, newW, newH);
    imagePyramids[downscaleFactor] = scaled;
    return scaled;
}

MTensor& Camera::getGPUImage(int downscaleFactor) {
    auto it = mtensorImageCache.find(downscaleFactor);
    if (it != mtensorImageCache.end()) return it->second;
    Image img = getImage(downscaleFactor);
    MTensor mt = gpu_empty({img.height, img.width, 3}, DType::Float32);
    memcpy(mt.data_ptr(), img.ptr(), img.width * img.height * 3 * sizeof(float));
    mtensorImageCache[downscaleFactor] = mt;
    return mtensorImageCache[downscaleFactor];
}

// ── Ground-truth optical flow (Phase 3) ─────────────────────────────────────
//
// Middlebury .flo: float magic 202021.25, int32 width, int32 height, then
// width*height interleaved (u, v) float pairs. Values >= 1e9 mean "unknown",
// which is how every estimator marks occlusions and out-of-frame pixels.
// We expand that to (u, v, valid) so the loss kernel can mask in one read.

MTensor* Camera::getGPUFlow() {
    if (gpuFlow.defined()) return &gpuFlow;
    if (flowPath.empty()) return nullptr;

    std::ifstream f(flowPath, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "Flow: cannot open %s\n", flowPath.c_str());
        flowPath.clear();
        return nullptr;
    }
    float magic = 0; int32_t w = 0, h = 0;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    f.read(reinterpret_cast<char*>(&w), sizeof(w));
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f || magic != 202021.25f || w <= 0 || h <= 0) {
        fprintf(stderr, "Flow: %s is not a Middlebury .flo file\n", flowPath.c_str());
        flowPath.clear();
        return nullptr;
    }
    if (w != width || h != height) {
        fprintf(stderr, "Flow: %s is %dx%d but the image is %dx%d — ignoring\n",
                flowPath.c_str(), w, h, width, height);
        flowPath.clear();
        return nullptr;
    }

    std::vector<float> uv((size_t)w * h * 2);
    f.read(reinterpret_cast<char*>(uv.data()), (std::streamsize)(uv.size() * sizeof(float)));
    if (!f) {
        fprintf(stderr, "Flow: %s is truncated\n", flowPath.c_str());
        flowPath.clear();
        return nullptr;
    }

    gpuFlow = gpu_empty({h, w, 3}, DType::Float32);
    float *dst = gpuFlow.data<float>();
    for (size_t i = 0; i < (size_t)w * h; i++) {
        float u = uv[i * 2], v = uv[i * 2 + 1];
        bool ok = std::abs(u) < 1e9f && std::abs(v) < 1e9f &&
                  std::isfinite(u) && std::isfinite(v);
        dst[i * 3 + 0] = ok ? u : 0.0f;
        dst[i * 3 + 1] = ok ? v : 0.0f;
        dst[i * 3 + 2] = ok ? 1.0f : 0.0f;
    }
    return &gpuFlow;
}

// Links every camera to the next one in time and looks for its ground-truth
// flow file under <dataset>/flow/<image stem>.flo. Returns the number of
// cameras that ended up with a flow target.
//
// Call this on the TRAIN camera list, after the train/test split — flowNextIdx
// indexes exactly the vector passed in, and a test frame must never be a flow
// target or the held-out views leak into training.
//
// The pairing is by timestamp, not by file order: a dataset whose frames are
// not stored in temporal order would otherwise get flow between unrelated
// frames.
int attachFlowToCameras(std::vector<Camera> &cams, const std::string &datasetPath) {
    fs::path flowDir = fs::path(datasetPath) / "flow";
    if (!fs::is_directory(flowDir)) return 0;

    std::vector<int> order(cams.size());
    for (size_t i = 0; i < cams.size(); i++) order[i] = (int)i;
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return cams[a].time < cams[b].time; });

    int linked = 0;
    for (size_t k = 0; k + 1 < order.size(); k++) {
        Camera &cur = cams[order[k]];
        float dt = cams[order[k + 1]].time - cur.time;
        if (dt <= 0.0f) continue;          // duplicate timestamps: no flow direction
        fs::path p = flowDir / (fs::path(cur.filePath).stem().string() + ".flo");
        if (!fs::exists(p)) continue;
        cur.flowPath = p.string();
        cur.flowNextIdx = order[k + 1];
        cur.flowDt = dt;
        linked++;
    }
    return linked;
}

// ── Scale & center ──────────────────────────────────────────────────────────

void autoScaleAndCenter(InputData &data) {
    if (data.cameras.empty()) return;

    // Compute mean camera position
    float mean[3] = {};
    for (auto &cam : data.cameras) {
        mean[0] += cam.camToWorld[3];   // column 3 of row 0
        mean[1] += cam.camToWorld[7];   // column 3 of row 1
        mean[2] += cam.camToWorld[11];  // column 3 of row 2
    }
    int n = (int)data.cameras.size();
    mean[0] /= n; mean[1] /= n; mean[2] /= n;

    data.translation[0] = mean[0];
    data.translation[1] = mean[1];
    data.translation[2] = mean[2];

    // Center camera poses
    for (auto &cam : data.cameras) {
        cam.camToWorld[3]  -= mean[0];
        cam.camToWorld[7]  -= mean[1];
        cam.camToWorld[11] -= mean[2];
    }

    // Compute scale from max absolute camera position
    float maxAbs = 0;
    for (auto &cam : data.cameras) {
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[3]));
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[7]));
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[11]));
    }
    data.scale = (maxAbs > 0) ? (1.0f / maxAbs) : 1.0f;

    // Apply scale to camera positions
    for (auto &cam : data.cameras) {
        cam.camToWorld[3]  *= data.scale;
        cam.camToWorld[7]  *= data.scale;
        cam.camToWorld[11] *= data.scale;
    }

    // Apply to point cloud
    for (int64_t i = 0; i < data.points.count; i++) {
        data.points.xyz[i*3+0] = (data.points.xyz[i*3+0] - mean[0]) * data.scale;
        data.points.xyz[i*3+1] = (data.points.xyz[i*3+1] - mean[1]) * data.scale;
        data.points.xyz[i*3+2] = (data.points.xyz[i*3+2] - mean[2]) * data.scale;
    }
}

// ── Train/test split ────────────────────────────────────────────────────────

std::tuple<std::vector<Camera>, Camera*> InputData::getCameras(bool validate, const std::string &valImage) {
    if (!validate) return {cameras, nullptr};

    // Find validation camera
    int valIdx = -1;
    if (valImage == "random") {
        std::mt19937 rng(42);
        valIdx = rng() % cameras.size();
    } else {
        for (int i = 0; i < (int)cameras.size(); i++) {
            if (cameras[i].filePath.find(valImage) != std::string::npos) { valIdx = i; break; }
        }
    }
    if (valIdx < 0) valIdx = 0;

    Camera *valCam = &cameras[valIdx];
    std::vector<Camera> train;
    for (int i = 0; i < (int)cameras.size(); i++)
        if (i != valIdx) train.push_back(cameras[i]);

    return {train, valCam};
}

std::tuple<std::vector<Camera>, std::vector<Camera>> InputData::splitTrainTest(int testEvery) {
    std::vector<Camera> train, test;
    // Datasets that ship their own split (D-NeRF) are honoured verbatim —
    // holding out every Nth frame there would leak test poses into training.
    if (hasExplicitSplit) {
        for (auto &cam : cameras) (cam.isTest ? test : train).push_back(cam);
        return {train, test};
    }
    for (int i = 0; i < (int)cameras.size(); i++) {
        if (i % testEvery == 0)
            test.push_back(cameras[i]);
        else
            train.push_back(cameras[i]);
    }
    return {train, test};
}

// ── Save cameras ────────────────────────────────────────────────────────────

void InputData::saveCameras(const std::string &filename, bool keepCrs) const {
    json arr = json::array();
    for (auto &cam : cameras) {
        json c;
        c["file_path"] = fs::path(cam.filePath).filename().string();
        c["width"] = cam.width;
        c["height"] = cam.height;
        c["fx"] = cam.fx; c["fy"] = cam.fy;
        c["cx"] = cam.cx; c["cy"] = cam.cy;

        // Extract rotation and translation from camToWorld
        float R[9], T[3];
        // Undo OpenGL flip (negate columns 1,2 back to OpenCV convention)
        R[0] =  cam.camToWorld[0]; R[1] = -cam.camToWorld[1]; R[2] = -cam.camToWorld[2];
        R[3] =  cam.camToWorld[4]; R[4] = -cam.camToWorld[5]; R[5] = -cam.camToWorld[6];
        R[6] =  cam.camToWorld[8]; R[7] = -cam.camToWorld[9]; R[8] = -cam.camToWorld[10];
        T[0] =  cam.camToWorld[3]; T[1] =  cam.camToWorld[7]; T[2] =  cam.camToWorld[11];

        if (keepCrs) {
            T[0] = T[0] / scale + translation[0];
            T[1] = T[1] / scale + translation[1];
            T[2] = T[2] / scale + translation[2];
        }

        c["rotation"] = {{R[0],R[1],R[2]},{R[3],R[4],R[5]},{R[6],R[7],R[8]}};
        c["translation"] = {T[0], T[1], T[2]};
        arr.push_back(c);
    }

    std::ofstream f(filename);
    f << arr.dump(2);
}

// ── Format dispatcher ───────────────────────────────────────────────────────

InputData inputDataFromX(const std::string &path, const std::string &colmapImagePath,
                         bool whiteBackground) {
    fs::path root(path);

    // D-NeRF: per-split transforms with a per-frame time. Checked before
    // Nerfstudio because some scenes ship both.
    if (fs::exists(root / "transforms_train.json"))
        return loaders::loadDnerf(path, whiteBackground);

    // Nerfstudio: transforms.json
    if (fs::exists(root / "transforms.json"))
        return loaders::loadNerfstudio(path);

    // COLMAP: cameras.bin (direct or in sparse/0/)
    if (fs::exists(root / "cameras.bin") || fs::exists(root / "sparse" / "0" / "cameras.bin"))
        return loaders::loadColmap(path, colmapImagePath);

    // Polycam: keyframes/ directory or cameras.json
    if (fs::exists(root / "keyframes" / "corrected_cameras") || fs::exists(root / "cameras.json"))
        return loaders::loadPolycam(path);

    throw std::runtime_error("Unrecognized dataset format in: " + path +
        "\nSupported: COLMAP (cameras.bin), Nerfstudio (transforms.json), "
        "D-NeRF (transforms_train.json), Polycam (keyframes/)");
}
