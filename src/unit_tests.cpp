#include "metrics.hpp"
#include "postprocessing.hpp"
#include "preprocessing.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

void expect(bool cond, const char* msg) {
    if (!cond) throw std::runtime_error(msg);
}

void expectApprox(double actual, double expected, double tol, const char* msg) {
    if (std::abs(actual - expected) > tol) {
        std::fprintf(stderr, "Assertion failed (%s): actual=%.6f expected=%.6f\n", msg, actual, expected);
        throw std::runtime_error(msg);
    }
}

void testComputeLetterbox() {
    const auto p1 = yolo::computeLetterbox(810, 1080, 640, 640);
    expect(p1.resized_w == 480, "p1 resized_w");
    expect(p1.resized_h == 640, "p1 resized_h");
    expect(p1.pad_x == 80, "p1 pad_x");
    expect(p1.pad_y == 0, "p1 pad_y");

    const auto p2 = yolo::computeLetterbox(1280, 720, 640, 640);
    expect(p2.resized_w == 640, "p2 resized_w");
    expect(p2.resized_h == 360, "p2 resized_h");
    expect(p2.pad_x == 0, "p2 pad_x");
    expect(p2.pad_y == 140, "p2 pad_y");
}

void testPreprocessRgbToNchw() {
    std::vector<uint8_t> src(4 * 2 * 3, 255);
    src[0] = 255;
    src[1] = 0;
    src[2] = 0;

    std::vector<float> out(3 * 4 * 4, 0.0f);
    const auto params = yolo::preprocessRgbToNchw(src.data(), 4, 2, 4, 4, out.data(), out.size());
    expect(params.pad_y == 1, "pad_y == 1");
    expectApprox(out[0], yolo::PAD_VALUE_F32, 1e-6, "padding value");
    expectApprox(out[1 * 4 + 0], 1.0f, 1e-5, "R=1.0");
    expectApprox(out[16 + 1 * 4 + 0], 0.0f, 1e-5, "G=0.0");
    expectApprox(out[32 + 1 * 4 + 0], 0.0f, 1e-5, "B=0.0");
}

void testInferOutputDescriptorAndDecode() {
    const auto d_e2e = yolo::inferOutputDescriptor({1, 300, 6});
    expect(d_e2e.format == yolo::OutputFormat::yolo26_end2end, "yolo26_end2end");
    expect(!yolo::formatRequiresNms(d_e2e.format), "end2end does not require NMS");

    const auto d_raw = yolo::inferOutputDescriptor({1, 84, 8400});
    expect(d_raw.format == yolo::OutputFormat::yolo_raw_ch_first, "yolo_raw_ch_first");
    expect(yolo::formatRequiresNms(d_raw.format), "raw requires NMS");

    const auto lb = yolo::computeLetterbox(810, 1080, 640, 640);
    const float raw[12] = {
        80.0f, 135.0f, 560.0f, 445.0f, 0.92f, 5.0f,
        10.0f, 10.0f,  50.0f,  50.0f,  0.10f, 0.0f,
    };
    const auto desc = yolo::inferOutputDescriptor({1, 2, 6});
    yolo::ScratchBuffer scratch(16);
    yolo::Detection out[16];
    const size_t n = yolo::decodeOutput(raw, desc, lb, 0.25f, 0.45f, false, scratch, out, 16);
    expect(n == 1, "1 detection");
    expect(out[0].class_id == 5, "bus class_id");
    expect(out[0].x1 >= 0.0f && out[0].x2 <= 810.0f, "x bounds");
    expect(out[0].y1 >= 0.0f && out[0].y2 <= 1080.0f, "y bounds");
}

void testComputeStats() {
    std::vector<int64_t> samples(100);
    for (size_t i = 0; i < 100; ++i) {
        samples[i] = static_cast<int64_t>((i + 1) * 1'000'000LL);
    }
    std::vector<int64_t> scratch(100);
    const auto st = yolo::computeStats(samples, scratch);
    expectApprox(st.min_ms, 1.0, 1e-6, "min_ms");
    expectApprox(st.max_ms, 100.0, 1e-6, "max_ms");
    expectApprox(st.mean_ms, 50.5, 1e-6, "mean_ms");
    expectApprox(st.median_ms, 50.5, 1e-6, "median_ms");
    expectApprox(st.p95_ms, 95.05, 1e-6, "p95_ms");
    expectApprox(st.p99_ms, 99.01, 1e-6, "p99_ms");
}

} // namespace

int main() {
    testComputeLetterbox();
    testPreprocessRgbToNchw();
    testInferOutputDescriptorAndDecode();
    testComputeStats();
    std::printf("All C++ unit tests passed (4/4)\n");
    return 0;
}
