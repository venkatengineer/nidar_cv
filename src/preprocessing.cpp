#include "preprocessing.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace yolo {

LetterboxParams computeLetterbox(uint32_t orig_w, uint32_t orig_h, uint32_t dst_w, uint32_t dst_h) {
    const float fw = static_cast<float>(orig_w);
    const float fh = static_cast<float>(orig_h);
    const float fdw = static_cast<float>(dst_w);
    const float fdh = static_cast<float>(dst_h);

    const float scale = std::min(fdw / fw, fdh / fh);
    const uint32_t resized_w = std::max<uint32_t>(1u, static_cast<uint32_t>(std::round(fw * scale)));
    const uint32_t resized_h = std::max<uint32_t>(1u, static_cast<uint32_t>(std::round(fh * scale)));

    const float dw = static_cast<float>(dst_w - resized_w) * 0.5f;
    const float dh = static_cast<float>(dst_h - resized_h) * 0.5f;
    const uint32_t pad_x = static_cast<uint32_t>(std::round(std::max(0.0f, dw - 0.1f)));
    const uint32_t pad_y = static_cast<uint32_t>(std::round(std::max(0.0f, dh - 0.1f)));

    return LetterboxParams{
        orig_w,
        orig_h,
        dst_w,
        dst_h,
        resized_w,
        resized_h,
        pad_x,
        pad_y,
        scale,
    };
}

LetterboxParams preprocessRgbToNchw(
    const uint8_t* rgb_data,
    uint32_t orig_w,
    uint32_t orig_h,
    uint32_t dst_w,
    uint32_t dst_h,
    float* out_tensor,
    size_t out_tensor_len) {
    const size_t plane_size = static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h);
    assert(out_tensor_len == plane_size * 3);
    (void)out_tensor_len;

    const LetterboxParams params = computeLetterbox(orig_w, orig_h, dst_w, dst_h);

    std::fill(out_tensor, out_tensor + plane_size * 3, PAD_VALUE_F32);

    float* r_plane = out_tensor;
    float* g_plane = out_tensor + plane_size;
    float* b_plane = out_tensor + plane_size * 2;

    const size_t src_stride = static_cast<size_t>(orig_w) * 3;
    const float inv_scale_x = static_cast<float>(orig_w) / static_cast<float>(params.resized_w);
    const float inv_scale_y = static_cast<float>(orig_h) / static_cast<float>(params.resized_h);
    constexpr float inv_255 = 1.0f / 255.0f;
    const float max_sx = static_cast<float>(orig_w - 1);
    const float max_sy = static_cast<float>(orig_h - 1);

    for (uint32_t dy = 0; dy < params.resized_h; ++dy) {
        const float sy_f = std::clamp(((static_cast<float>(dy) + 0.5f) * inv_scale_y) - 0.5f, 0.0f, max_sy);
        const uint32_t y0 = static_cast<uint32_t>(std::floor(sy_f));
        const uint32_t y1 = std::min(y0 + 1, orig_h - 1);
        const float wy = sy_f - static_cast<float>(y0);
        const float wy_inv = 1.0f - wy;

        const size_t row0_off = static_cast<size_t>(y0) * src_stride;
        const size_t row1_off = static_cast<size_t>(y1) * src_stride;
        const size_t dst_row_off =
            static_cast<size_t>(params.pad_y + dy) * static_cast<size_t>(dst_w) +
            static_cast<size_t>(params.pad_x);

        for (uint32_t dx = 0; dx < params.resized_w; ++dx) {
            const float sx_f = std::clamp(((static_cast<float>(dx) + 0.5f) * inv_scale_x) - 0.5f, 0.0f, max_sx);
            const uint32_t x0 = static_cast<uint32_t>(std::floor(sx_f));
            const uint32_t x1 = std::min(x0 + 1, orig_w - 1);
            const float wx = sx_f - static_cast<float>(x0);
            const float wx_inv = 1.0f - wx;

            const float w00 = wy_inv * wx_inv * inv_255;
            const float w01 = wy_inv * wx * inv_255;
            const float w10 = wy * wx_inv * inv_255;
            const float w11 = wy * wx * inv_255;

            const size_t p00 = row0_off + static_cast<size_t>(x0) * 3;
            const size_t p01 = row0_off + static_cast<size_t>(x1) * 3;
            const size_t p10 = row1_off + static_cast<size_t>(x0) * 3;
            const size_t p11 = row1_off + static_cast<size_t>(x1) * 3;

            const size_t dst_idx = dst_row_off + static_cast<size_t>(dx);
            r_plane[dst_idx] =
                static_cast<float>(rgb_data[p00]) * w00 +
                static_cast<float>(rgb_data[p01]) * w01 +
                static_cast<float>(rgb_data[p10]) * w10 +
                static_cast<float>(rgb_data[p11]) * w11;
            g_plane[dst_idx] =
                static_cast<float>(rgb_data[p00 + 1]) * w00 +
                static_cast<float>(rgb_data[p01 + 1]) * w01 +
                static_cast<float>(rgb_data[p10 + 1]) * w10 +
                static_cast<float>(rgb_data[p11 + 1]) * w11;
            b_plane[dst_idx] =
                static_cast<float>(rgb_data[p00 + 2]) * w00 +
                static_cast<float>(rgb_data[p01 + 2]) * w01 +
                static_cast<float>(rgb_data[p10 + 2]) * w10 +
                static_cast<float>(rgb_data[p11 + 2]) * w11;
        }
    }

    return params;
}

} // namespace yolo
