#pragma once

#include "onnxruntime_c_api.h"
#include "postprocessing.hpp"
#include "preprocessing.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace yolo {

struct DetectorConfig {
    std::string model_path;
    uint32_t intra_op_threads = 4;
    uint32_t inter_op_threads = 1;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    bool force_nms = false;
};

/// ONNX Runtime C API wrapper with preallocated zero-copy I/O tensors.
/// Mirrors the Zig Detector architecture 1:1.
class Detector {
public:
    explicit Detector(const DetectorConfig& config);
    ~Detector();

    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    LetterboxParams preprocess(const uint8_t* rgb_data, uint32_t width, uint32_t height);
    void inferRaw();
    size_t postprocess(const LetterboxParams& lb);

    const std::string& ortVersion() const { return ort_version_; }
    const std::array<int64_t, 4>& inputShape() const { return input_shape_; }
    const std::array<int64_t, 3>& outputShape() const { return output_shape_; }
    const OutputDescriptor& outputDesc() const { return output_desc_; }
    const Detection* detections() const { return detections_buf_.data(); }

private:
    DetectorConfig config_;
    const OrtApi* api_ = nullptr;
    std::string ort_version_;
    OrtEnv* env_ = nullptr;
    OrtSessionOptions* session_options_ = nullptr;
    OrtSession* session_ = nullptr;
    OrtMemoryInfo* memory_info_ = nullptr;
    OrtAllocator* ort_allocator_ = nullptr;

    std::string input_name_;
    std::string output_name_;
    std::array<int64_t, 4> input_shape_{};
    std::array<int64_t, 3> output_shape_{};
    OutputDescriptor output_desc_{};

    std::vector<float> input_tensor_buf_;
    std::vector<float> output_tensor_buf_;
    OrtValue* input_ort_value_ = nullptr;
    OrtValue* output_ort_value_ = nullptr;

    ScratchBuffer scratch_;
    std::vector<Detection> detections_buf_;

    void cleanup() noexcept;
    void checkStatus(OrtStatus* status) const;
    std::array<int64_t, 4> validateInputTensor() const;
    std::array<int64_t, 3> validateOutputTensor() const;
};

} // namespace yolo
