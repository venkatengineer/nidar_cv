#pragma once

#include <cstddef>
#include <cstdint>

namespace yolo {

inline constexpr uint32_t INPUT_WIDTH = 640;
inline constexpr uint32_t INPUT_HEIGHT = 640;
inline constexpr uint32_t INPUT_CHANNELS = 3;
inline constexpr size_t INPUT_TENSOR_ELEMENTS =
    static_cast<size_t>(INPUT_CHANNELS) * INPUT_HEIGHT * INPUT_WIDTH;
inline constexpr uint8_t PAD_VALUE_U8 = 114;
inline constexpr float PAD_VALUE_F32 = static_cast<float>(PAD_VALUE_U8) / 255.0f;

struct LetterboxParams {
    uint32_t orig_w;
    uint32_t orig_h;
    uint32_t dst_w;
    uint32_t dst_h;
    uint32_t resized_w;
    uint32_t resized_h;
    uint32_t pad_x;
    uint32_t pad_y;
    float scale;
};

LetterboxParams computeLetterbox(uint32_t orig_w, uint32_t orig_h, uint32_t dst_w, uint32_t dst_h);

/// Deterministic preprocessing pipeline matching the Zig baseline:
/// 1. Aspect-ratio-preserving letterbox to dst_w x dst_h with pad = 114/255.
/// 2. OpenCV INTER_LINEAR pixel-center aligned bilinear resize.
/// 3. RGB channel ordering, FP32 normalization [0, 255] -> [0.0, 1.0], NCHW layout.
/// Writes directly into caller-owned `out_tensor` (zero heap allocations).
LetterboxParams preprocessRgbToNchw(
    const uint8_t* rgb_data,
    uint32_t orig_w,
    uint32_t orig_h,
    uint32_t dst_w,
    uint32_t dst_h,
    float* out_tensor,
    size_t out_tensor_len);

} // namespace yolo
