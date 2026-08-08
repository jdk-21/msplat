#ifndef LOADERS_H
#define LOADERS_H

#include "input_data.hpp"

// Format-specific loaders
namespace loaders {
    InputData loadColmap(const std::string &projectRoot, const std::string &imageSourcePath = "");
    InputData loadNerfstudio(const std::string &projectRoot);
    InputData loadPolycam(const std::string &projectRoot);
    // D-NeRF / Blender-style dynamic scenes: transforms_{train,val,test}.json
    // with a per-frame `time`. whiteBackground picks the compositing background
    // for the transparent PNGs (the published D-NeRF protocol uses white).
    InputData loadDnerf(const std::string &projectRoot, bool whiteBackground);
}

// PLY point cloud reader
Points readPly(const std::string &path);

// COLMAP binary point cloud reader
Points readColmapPoints(const std::string &path);

// Image I/O
// Returns float32 [0,1]. bg (RGB, optional) composites source alpha over that
// colour instead of discarding it.
Image imreadRGB(const std::string &path, const float *bg = nullptr);
Image resizeArea(const Image &src, int dstW, int dstH);  // box-filter downscale
void imwriteRGB(const std::string &path, const Image &img);  // save as PNG

// Undistortion (Brown-Conrady model, alpha=0 crop)
struct UndistortResult {
    Image image;
    float fx, fy, cx, cy;  // updated intrinsics after crop
    int width, height;      // cropped dimensions
};
UndistortResult undistortImage(const Image &src,
    float fx, float fy, float cx, float cy,
    float k1, float k2, float p1, float p2, float k3);

// Pose utilities
void autoScaleAndCenter(InputData &data);

// Gaussian PLY/splat I/O (trained scene export/import)
struct GaussianParams {
    MTensor &means, &scales, &quats, &featuresDc, &featuresRest, &opacities;
    float scale;          // CRS scale factor
    float translation[3]; // CRS translation
    bool keepCrs;
};

// maxShBases caps how many SH rest bases are written; -1 writes all of them.
// 0 yields a DC-only file — same geometry and count, roughly a quarter of the
// size, which is what a preview frame sequence wants.
void saveGaussianPly(const std::string &path, GaussianParams &p, int step, int maxShBases = -1);
void saveGaussianSplat(const std::string &path, GaussianParams &p);

struct LoadedGaussians {
    MTensor means, scales, quats, featuresDc, featuresRest, opacities;
    int step;
};
LoadedGaussians loadGaussianPly(const std::string &path, float scale, const float translation[3], bool keepCrs);

#endif
