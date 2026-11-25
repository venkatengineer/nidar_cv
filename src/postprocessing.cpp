#include "postprocessing.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace yolo {

namespace {

constexpr std::array<const char*, NUM_COCO_CLASSES> COCO_CLASSES = {
    "person",        "bicycle",      "car",           "motorcycle",    "airplane",
    "bus",           "train",        "truck",         "boat",          "traffic light",
    "fire hydrant",  "stop sign",    "parking meter", "bench",         "bird",
    "cat",           "dog",          "horse",         "sheep",         "cow",
    "elephant",      "bear",         "zebra",         "giraffe",       "backpack",
    "umbrella",      "handbag",      "tie",           "suitcase",      "frisbee",
    "skis",          "snowboard",    "sports ball",   "kite",          "baseball bat",
    "baseball glove","skateboard",   "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",          "fork",          "knife",         "spoon",
    "bowl",          "banana",       "apple",         "sandwich",      "orange",
    "broccoli",      "carrot",       "hot dog",       "pizza",         "donut",
    "cake",          "chair",        "couch",         "potted plant",  "bed",
    "dining table",  "toilet",       "tv",            "laptop",        "mouse",
    "remote",        "keyboard",     "cell phone",    "microwave",     "oven",
    "toaster",       "sink",         "refrigerator",  "book",          "clock",
    "vase",          "scissors",     "teddy bear",    "hair drier",    "toothbrush",
};

size_t decodeEnd2End(
    const float* raw_output,
    size_t num_rows,
    const LetterboxParams& lb,
    float conf_threshold,
    Detection* out,
    size_t out_cap) {
    size_t n = 0;
    const float inv_scale = 1.0f / lb.scale;
    const float fpad_x = static_cast<float>(lb.pad_x);
    const float fpad_y = static_cast<float>(lb.pad_y);
    const float max_w = static_cast<float>(lb.orig_w);
    const float max_h = static_cast<float>(lb.orig_h);

    for (size_t i = 0; i < num_rows; ++i) {
        const size_t base = i * 6;
        const float score = raw_output[base + 4];
        if (!std::isfinite(score) || score < conf_threshold) continue;

        const float cls_f = raw_output[base + 5];
        if (!std::isfinite(cls_f) || cls_f < 0.0f) continue;
        const uint32_t cls_id = static_cast<uint32_t>(std::round(cls_f));
        if (cls_id >= NUM_COCO_CLASSES) continue;

        const float x1 = std::clamp((raw_output[base + 0] - fpad_x) * inv_scale, 0.0f, max_w);
        const float y1 = std::clamp((raw_output[base + 1] - fpad_y) * inv_scale, 0.0f, max_h);
        const float x2 = std::clamp((raw_output[base + 2] - fpad_x) * inv_scale, 0.0f, max_w);
        const float y2 = std::clamp((raw_output[base + 3] - fpad_y) * inv_scale, 0.0f, max_h);
        if (x2 <= x1 || y2 <= y1) continue;

        if (n < out_cap) {
            out[n++] = Detection{
                x1,
                y1,
                x2,
                y2,
                std::clamp(score, 0.0f, 1.0f),
                cls_id,
            };
        }
    }
    return n;
}

size_t decodeRawChannelFirst(
    const float* raw_output,
    size_t num_anchors,
    size_t num_classes,
    const LetterboxParams& lb,
    float conf_threshold,
    Detection* out,
    size_t out_cap) {
    size_t n = 0;
    const float inv_scale = 1.0f / lb.scale;
    const float fpad_x = static_cast<float>(lb.pad_x);
    const float fpad_y = static_cast<float>(lb.pad_y);
    const float max_w = static_cast<float>(lb.orig_w);
    const float max_h = static_cast<float>(lb.orig_h);

    for (size_t a = 0; a < num_anchors; ++a) {
        float best_score = -1.0f;
        uint32_t best_cls = 0;
        for (size_t c = 0; c < num_classes; ++c) {
            const float s = raw_output[(4 + c) * num_anchors + a];
            if (s > best_score) {
                best_score = s;
                best_cls = static_cast<uint32_t>(c);
            }
        }
        if (!std::isfinite(best_score) || best_score < conf_threshold) continue;

        const float cx = raw_output[0 * num_anchors + a];
        const float cy = raw_output[1 * num_anchors + a];
        const float w = raw_output[2 * num_anchors + a];
        const float h = raw_output[3 * num_anchors + a];
        const float half_w = w * 0.5f;
        const float half_h = h * 0.5f;

        const float x1 = std::clamp(((cx - half_w) - fpad_x) * inv_scale, 0.0f, max_w);
        const float y1 = std::clamp(((cy - half_h) - fpad_y) * inv_scale, 0.0f, max_h);
        const float x2 = std::clamp(((cx + half_w) - fpad_x) * inv_scale, 0.0f, max_w);
        const float y2 = std::clamp(((cy + half_h) - fpad_y) * inv_scale, 0.0f, max_h);
        if (x2 <= x1 || y2 <= y1) continue;

        if (n < out_cap) {
            out[n++] = Detection{
                x1,
                y1,
                x2,
                y2,
                std::clamp(best_score, 0.0f, 1.0f),
                best_cls,
            };
        }
    }
    return n;
}

size_t decodeRawChannelLast(
    const float* raw_output,
    size_t num_anchors,
    size_t num_classes,
    const LetterboxParams& lb,
    float conf_threshold,
    Detection* out,
    size_t out_cap) {
    size_t n = 0;
    const size_t stride = 4 + num_classes;
    const float inv_scale = 1.0f / lb.scale;
    const float fpad_x = static_cast<float>(lb.pad_x);
    const float fpad_y = static_cast<float>(lb.pad_y);
    const float max_w = static_cast<float>(lb.orig_w);
    const float max_h = static_cast<float>(lb.orig_h);

    for (size_t a = 0; a < num_anchors; ++a) {
        const size_t base = a * stride;
        float best_score = -1.0f;
        uint32_t best_cls = 0;
        for (size_t c = 0; c < num_classes; ++c) {
            const float s = raw_output[base + 4 + c];
            if (s > best_score) {
                best_score = s;
                best_cls = static_cast<uint32_t>(c);
            }
        }
        if (!std::isfinite(best_score) || best_score < conf_threshold) continue;

        const float cx = raw_output[base + 0];
        const float cy = raw_output[base + 1];
        const float w = raw_output[base + 2];
        const float h = raw_output[base + 3];
        const float half_w = w * 0.5f;
        const float half_h = h * 0.5f;

        const float x1 = std::clamp(((cx - half_w) - fpad_x) * inv_scale, 0.0f, max_w);
        const float y1 = std::clamp(((cy - half_h) - fpad_y) * inv_scale, 0.0f, max_h);
        const float x2 = std::clamp(((cx + half_w) - fpad_x) * inv_scale, 0.0f, max_w);
        const float y2 = std::clamp(((cy + half_h) - fpad_y) * inv_scale, 0.0f, max_h);
        if (x2 <= x1 || y2 <= y1) continue;

        if (n < out_cap) {
            out[n++] = Detection{
                x1,
                y1,
                x2,
                y2,
                std::clamp(best_score, 0.0f, 1.0f),
                best_cls,
            };
        }
    }
    return n;
}

float computeIou(const Detection& a, const Detection& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;
    if (inter <= 0.0f) return 0.0f;

    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    const float union_area = area_a + area_b - inter;
    if (union_area <= 0.0f) return 0.0f;
    return inter / union_area;
}

size_t applyClassAwareNms(
    const Detection* sorted_candidates,
    size_t cand_count,
    uint8_t* suppressed,
    float iou_threshold,
    Detection* out_detections,
    size_t out_cap) {
    std::memset(suppressed, 0, cand_count);
    size_t out_count = 0;

    for (size_t i = 0; i < cand_count; ++i) {
        if (suppressed[i]) continue;
        if (out_count >= out_cap) break;

        const Detection& cand = sorted_candidates[i];
        out_detections[out_count++] = cand;

        for (size_t j = i + 1; j < cand_count; ++j) {
            if (suppressed[j]) continue;
            if (sorted_candidates[j].class_id != cand.class_id) continue;
            if (computeIou(cand, sorted_candidates[j]) > iou_threshold) {
                suppressed[j] = 1;
            }
        }
    }
    return out_count;
}

} // namespace

std::string_view className(uint32_t class_id) {
    if (class_id < NUM_COCO_CLASSES) return COCO_CLASSES[class_id];
    return "unknown";
}

OutputDescriptor inferOutputDescriptor(const std::array<int64_t, 3>& shape) {
    if (shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0) {
        throw std::runtime_error("Unsupported ONNX output tensor shape");
    }
    const size_t d1 = static_cast<size_t>(shape[1]);
    const size_t d2 = static_cast<size_t>(shape[2]);

    if (d2 == 6) {
        return OutputDescriptor{OutputFormat::yolo26_end2end, d1, d2, NUM_COCO_CLASSES};
    }
    if (d1 == 4 + NUM_COCO_CLASSES && d2 > 6) {
        return OutputDescriptor{OutputFormat::yolo_raw_ch_first, d1, d2, NUM_COCO_CLASSES};
    }
    if (d2 == 4 + NUM_COCO_CLASSES && d1 > 6) {
        return OutputDescriptor{OutputFormat::yolo_raw_ch_last, d1, d2, NUM_COCO_CLASSES};
    }
    throw std::runtime_error("Unsupported ONNX output tensor dimensions");
}

size_t decodeOutput(
    const float* raw_output,
    const OutputDescriptor& desc,
    const LetterboxParams& lb,
    float conf_threshold,
    float iou_threshold,
    bool force_nms,
    ScratchBuffer& scratch,
    Detection* out_detections,
    size_t out_capacity) {
    size_t count = 0;
    switch (desc.format) {
        case OutputFormat::yolo26_end2end:
            count = decodeEnd2End(
                raw_output,
                desc.dim1,
                lb,
                conf_threshold,
                force_nms ? scratch.candidates.data() : out_detections,
                force_nms ? scratch.candidates.size() : out_capacity);
            break;
        case OutputFormat::yolo_raw_ch_first:
            count = decodeRawChannelFirst(
                raw_output,
                desc.dim2,
                desc.num_classes,
                lb,
                conf_threshold,
                scratch.candidates.data(),
                scratch.candidates.size());
            break;
        case OutputFormat::yolo_raw_ch_last:
            count = decodeRawChannelLast(
                raw_output,
                desc.dim1,
                desc.num_classes,
                lb,
                conf_threshold,
                scratch.candidates.data(),
                scratch.candidates.size());
            break;
    }

    auto desc_cmp = [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    };

    const bool run_nms = formatRequiresNms(desc.format) || force_nms;
    if (!run_nms) {
        std::sort(out_detections, out_detections + count, desc_cmp);
        return count;
    }

    std::sort(scratch.candidates.data(), scratch.candidates.data() + count, desc_cmp);
    return applyClassAwareNms(
        scratch.candidates.data(),
        count,
        scratch.suppressed.data(),
        iou_threshold,
        out_detections,
        out_capacity);
}

} // namespace yolo
