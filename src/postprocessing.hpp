#pragma once

#include "preprocessing.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace yolo {

inline constexpr size_t NUM_COCO_CLASSES = 80;
inline constexpr size_t MAX_DETECTIONS = 300;

std::string_view className(uint32_t class_id);

struct Detection {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    uint32_t class_id;
};

enum class OutputFormat {
    yolo26_end2end,
    yolo_raw_ch_first,
    yolo_raw_ch_last,
};

inline bool formatRequiresNms(OutputFormat fmt) {
    return fmt != OutputFormat::yolo26_end2end;
}

inline const char* outputFormatName(OutputFormat fmt) {
    switch (fmt) {
        case OutputFormat::yolo26_end2end:
            return "yolo26_end2end";
        case OutputFormat::yolo_raw_ch_first:
            return "yolo_raw_ch_first";
        case OutputFormat::yolo_raw_ch_last:
            return "yolo_raw_ch_last";
    }
    return "unknown";
}

struct OutputDescriptor {
    OutputFormat format;
    size_t dim1;
    size_t dim2;
    size_t num_classes;
};

OutputDescriptor inferOutputDescriptor(const std::array<int64_t, 3>& shape);

/// Preallocated reusable scratch buffers for NMS when decoding raw outputs.
struct ScratchBuffer {
    std::vector<Detection> candidates;
    std::vector<uint8_t> suppressed;

    explicit ScratchBuffer(size_t max_candidates)
        : candidates(max_candidates), suppressed(max_candidates, 0) {}
};

/// Decodes raw FP32 ONNX output into `out_detections` using zero heap allocations.
/// Returns the number of valid detections written, sorted by descending confidence.
size_t decodeOutput(
    const float* raw_output,
    const OutputDescriptor& desc,
    const LetterboxParams& lb,
    float conf_threshold,
    float iou_threshold,
    bool force_nms,
    ScratchBuffer& scratch,
    Detection* out_detections,
    size_t out_capacity);

} // namespace yolo
