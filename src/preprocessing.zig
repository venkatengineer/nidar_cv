const std = @import("std");

pub const INPUT_WIDTH: u32 = 640;
pub const INPUT_HEIGHT: u32 = 640;
pub const INPUT_CHANNELS: u32 = 3;
pub const INPUT_TENSOR_ELEMENTS: usize = INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;
pub const PAD_VALUE_U8: u8 = 114;
pub const PAD_VALUE_F32: f32 = @as(f32, PAD_VALUE_U8) / 255.0;

/// Metadata produced by letterbox preprocessing, used during postprocessing
/// to map 640x640 model coordinates back to original image pixel space.
pub const LetterboxParams = struct {
    orig_w: u32,
    orig_h: u32,
    dst_w: u32,
    dst_h: u32,
    resized_w: u32,
    resized_h: u32,
    pad_x: u32,
    pad_y: u32,
    scale: f32,
};

pub fn computeLetterbox(orig_w: u32, orig_h: u32, dst_w: u32, dst_h: u32) LetterboxParams {
    const fw: f32 = @floatFromInt(orig_w);
    const fh: f32 = @floatFromInt(orig_h);
    const fdw: f32 = @floatFromInt(dst_w);
    const fdh: f32 = @floatFromInt(dst_h);

    const scale = @min(fdw / fw, fdh / fh);
    const resized_w: u32 = @max(1, @as(u32, @intFromFloat(@round(fw * scale))));
    const resized_h: u32 = @max(1, @as(u32, @intFromFloat(@round(fh * scale))));

    // Match Ultralytics rounding: dw = (dst_w - resized_w) / 2 -> left pad = round(dw - 0.1)
    const dw: f32 = @as(f32, @floatFromInt(dst_w - resized_w)) * 0.5;
    const dh: f32 = @as(f32, @floatFromInt(dst_h - resized_h)) * 0.5;
    const pad_x: u32 = @intFromFloat(@round(@max(0.0, dw - 0.1)));
    const pad_y: u32 = @intFromFloat(@round(@max(0.0, dh - 0.1)));

    return .{
        .orig_w = orig_w,
        .orig_h = orig_h,
        .dst_w = dst_w,
        .dst_h = dst_h,
        .resized_w = resized_w,
        .resized_h = resized_h,
        .pad_x = pad_x,
        .pad_y = pad_y,
        .scale = scale,
    };
}

/// Fixed deterministic preprocessing pipeline (shared specification across languages):
/// 1. Aspect-ratio-preserving letterbox to dst_w x dst_h (640x640) with pad=114/255.
/// 2. Bilinear resize matching OpenCV INTER_LINEAR pixel-center alignment.
/// 3. RGB channel ordering, FP32 normalization [0, 255] -> [0.0, 1.0], NCHW layout.
///
/// Writes directly into caller-owned `out_tensor` (zero heap allocations).
pub fn preprocessRgbToNchw(
    rgb_data: []const u8,
    orig_w: u32,
    orig_h: u32,
    dst_w: u32,
    dst_h: u32,
    out_tensor: []f32,
) LetterboxParams {
    const plane_size: usize = @as(usize, dst_w) * @as(usize, dst_h);
    std.debug.assert(out_tensor.len == plane_size * 3);
    std.debug.assert(rgb_data.len == @as(usize, orig_w) * @as(usize, orig_h) * 3);

    const params = computeLetterbox(orig_w, orig_h, dst_w, dst_h);

    // Fill entire NCHW tensor with normalized pad value (114 / 255)
    @memset(out_tensor, PAD_VALUE_F32);

    const r_plane = out_tensor[0..plane_size];
    const g_plane = out_tensor[plane_size .. plane_size * 2];
    const b_plane = out_tensor[plane_size * 2 .. plane_size * 3];

    const src_stride: usize = @as(usize, orig_w) * 3;
    const inv_scale_x: f32 = @as(f32, @floatFromInt(orig_w)) / @as(f32, @floatFromInt(params.resized_w));
    const inv_scale_y: f32 = @as(f32, @floatFromInt(orig_h)) / @as(f32, @floatFromInt(params.resized_h));
    const inv_255: f32 = 1.0 / 255.0;
    const max_sx: f32 = @floatFromInt(orig_w - 1);
    const max_sy: f32 = @floatFromInt(orig_h - 1);

    var dy: u32 = 0;
    while (dy < params.resized_h) : (dy += 1) {
        const sy_f = std.math.clamp(((@as(f32, @floatFromInt(dy)) + 0.5) * inv_scale_y) - 0.5, 0.0, max_sy);
        const y0: u32 = @intFromFloat(@floor(sy_f));
        const y1: u32 = @min(y0 + 1, orig_h - 1);
        const wy: f32 = sy_f - @as(f32, @floatFromInt(y0));
        const wy_inv: f32 = 1.0 - wy;

        const row0_off = @as(usize, y0) * src_stride;
        const row1_off = @as(usize, y1) * src_stride;
        const dst_row_off = @as(usize, params.pad_y + dy) * @as(usize, dst_w) + @as(usize, params.pad_x);

        var dx: u32 = 0;
        while (dx < params.resized_w) : (dx += 1) {
            const sx_f = std.math.clamp(((@as(f32, @floatFromInt(dx)) + 0.5) * inv_scale_x) - 0.5, 0.0, max_sx);
            const x0: u32 = @intFromFloat(@floor(sx_f));
            const x1: u32 = @min(x0 + 1, orig_w - 1);
            const wx: f32 = sx_f - @as(f32, @floatFromInt(x0));
            const wx_inv: f32 = 1.0 - wx;

            const w00 = wy_inv * wx_inv * inv_255;
            const w01 = wy_inv * wx * inv_255;
            const w10 = wy * wx_inv * inv_255;
            const w11 = wy * wx * inv_255;

            const p00 = row0_off + @as(usize, x0) * 3;
            const p01 = row0_off + @as(usize, x1) * 3;
            const p10 = row1_off + @as(usize, x0) * 3;
            const p11 = row1_off + @as(usize, x1) * 3;

            const dst_idx = dst_row_off + @as(usize, dx);
            r_plane[dst_idx] =
                @as(f32, @floatFromInt(rgb_data[p00])) * w00 +
                @as(f32, @floatFromInt(rgb_data[p01])) * w01 +
                @as(f32, @floatFromInt(rgb_data[p10])) * w10 +
                @as(f32, @floatFromInt(rgb_data[p11])) * w11;
            g_plane[dst_idx] =
                @as(f32, @floatFromInt(rgb_data[p00 + 1])) * w00 +
                @as(f32, @floatFromInt(rgb_data[p01 + 1])) * w01 +
                @as(f32, @floatFromInt(rgb_data[p10 + 1])) * w10 +
                @as(f32, @floatFromInt(rgb_data[p11 + 1])) * w11;
            b_plane[dst_idx] =
                @as(f32, @floatFromInt(rgb_data[p00 + 2])) * w00 +
                @as(f32, @floatFromInt(rgb_data[p01 + 2])) * w01 +
                @as(f32, @floatFromInt(rgb_data[p10 + 2])) * w10 +
                @as(f32, @floatFromInt(rgb_data[p11 + 2])) * w11;
        }
    }

    return params;
}

test "computeLetterbox preserves aspect ratio for portrait and landscape" {
    const p1 = computeLetterbox(810, 1080, 640, 640);
    try std.testing.expectEqual(@as(u32, 480), p1.resized_w);
    try std.testing.expectEqual(@as(u32, 640), p1.resized_h);
    try std.testing.expectEqual(@as(u32, 80), p1.pad_x);
    try std.testing.expectEqual(@as(u32, 0), p1.pad_y);

    const p2 = computeLetterbox(1280, 720, 640, 640);
    try std.testing.expectEqual(@as(u32, 640), p2.resized_w);
    try std.testing.expectEqual(@as(u32, 360), p2.resized_h);
    try std.testing.expectEqual(@as(u32, 0), p2.pad_x);
    try std.testing.expectEqual(@as(u32, 140), p2.pad_y);
}

test "preprocessRgbToNchw normalizes and pads deterministically" {
    var src: [4 * 2 * 3]u8 = @splat(255);
    // Top-left pixel pure red (255, 0, 0)
    src[0] = 255;
    src[1] = 0;
    src[2] = 0;

    var out: [3 * 4 * 4]f32 = undefined;
    const params = preprocessRgbToNchw(&src, 4, 2, 4, 4, &out);
    try std.testing.expectEqual(@as(u32, 1), params.pad_y);
    // Row 0 is padding
    try std.testing.expectApproxEqAbs(PAD_VALUE_F32, out[0], 1e-6);
    // Row 1 col 0 is R=1.0, G=0.0, B=0.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[1 * 4 + 0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[16 + 1 * 4 + 0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[32 + 1 * 4 + 0], 1e-5);
}
