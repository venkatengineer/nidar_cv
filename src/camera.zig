const std = @import("std");
const cam_c = @import("camera_c");
const stb_c = @import("stb_c");

pub const PixelFormat = enum {
    yuyv,
    mjpg,
};

pub const CameraError = error{
    OpenFailed,
    FrameTimeout,
    CaptureFailed,
    DecodeFailed,
    OutOfMemory,
};

/// Native Linux V4L2 mmap streaming camera with reusable RGB24 output buffer.
pub const Camera = struct {
    allocator: std.mem.Allocator,
    raw: cam_c.V4L2Camera,
    width: u32,
    height: u32,
    pixfmt: PixelFormat,
    rgb_buf: []u8,

    pub fn open(
        allocator: std.mem.Allocator,
        device_path: []const u8,
        req_width: u32,
        req_height: u32,
        req_fmt: PixelFormat,
    ) CameraError!Camera {
        const dev_z = allocator.dupeSentinel(u8, device_path, 0) catch return CameraError.OutOfMemory;
        defer allocator.free(dev_z);

        var raw: cam_c.V4L2Camera = undefined;
        const c_fmt: cam_c.CamPixFmt = switch (req_fmt) {
            .yuyv => cam_c.CAM_PIXFMT_YUYV,
            .mjpg => cam_c.CAM_PIXFMT_MJPG,
        };
        const rc = cam_c.v4l2_cam_open(&raw, dev_z.ptr, req_width, req_height, c_fmt);
        if (rc != 0) {
            std.log.err("Failed to open V4L2 camera '{s}' (code={d})", .{ device_path, rc });
            return CameraError.OpenFailed;
        }
        errdefer cam_c.v4l2_cam_close(&raw);

        const actual_w: u32 = raw.width;
        const actual_h: u32 = raw.height;
        const actual_fmt: PixelFormat = if (raw.pixfmt == cam_c.CAM_PIXFMT_MJPG) .mjpg else .yuyv;

        const rgb_len: usize = @as(usize, actual_w) * @as(usize, actual_h) * 3;
        const rgb_buf = allocator.alloc(u8, rgb_len) catch return CameraError.OutOfMemory;

        return .{
            .allocator = allocator,
            .raw = raw,
            .width = actual_w,
            .height = actual_h,
            .pixfmt = actual_fmt,
            .rgb_buf = rgb_buf,
        };
    }

    pub fn close(self: *Camera) void {
        cam_c.v4l2_cam_close(&self.raw);
        self.allocator.free(self.rgb_buf);
        self.rgb_buf = &.{};
    }

    /// Captures one frame and writes RGB24 pixels into `self.rgb_buf`.
    /// In YUYV mode this performs zero heap allocations.
    pub fn captureRgb(self: *Camera, timeout_ms: i32) CameraError![]const u8 {
        var frame: cam_c.V4L2Frame = undefined;
        const rc = cam_c.v4l2_cam_dequeue(&self.raw, &frame, timeout_ms);
        if (rc == 1) return CameraError.FrameTimeout;
        if (rc < 0) return CameraError.CaptureFailed;
        defer _ = cam_c.v4l2_cam_enqueue(&self.raw, frame.index);

        const frame_bytes = frame.data[0..frame.size];
        switch (self.pixfmt) {
            .yuyv => yuyvToRgb24(frame_bytes, self.width, self.height, self.rgb_buf),
            .mjpg => {
                var w: c_int = 0;
                var h: c_int = 0;
                var comp: c_int = 0;
                const ptr = stb_c.stbi_load_from_memory(
                    frame_bytes.ptr,
                    @intCast(frame_bytes.len),
                    &w,
                    &h,
                    &comp,
                    3,
                );
                if (ptr == null or w <= 0 or h <= 0) return CameraError.DecodeFailed;
                defer stb_c.stbi_image_free(ptr);

                const expected_len = @as(usize, self.width) * @as(usize, self.height) * 3;
                const decoded_len = @as(usize, @intCast(w)) * @as(usize, @intCast(h)) * 3;
                if (decoded_len != expected_len) return CameraError.DecodeFailed;
                @memcpy(self.rgb_buf, ptr[0..decoded_len]);
            },
        }
        return self.rgb_buf;
    }
};

/// ITU-R BT.601 fixed-point YUYV (4:2:2 packed) -> RGB24 conversion.
pub fn yuyvToRgb24(yuyv: []const u8, width: u32, height: u32, out_rgb: []u8) void {
    const pixel_pairs = (@as(usize, width) * @as(usize, height)) / 2;
    const expected_in = pixel_pairs * 4;
    if (yuyv.len < expected_in or out_rgb.len < pixel_pairs * 6) return;

    var i: usize = 0;
    var out_i: usize = 0;
    while (i < expected_in) : ({
        i += 4;
        out_i += 6;
    }) {
        const y0: i32 = @intCast(yuyv[i + 0]);
        const u: i32 = @as(i32, yuyv[i + 1]) - 128;
        const y1: i32 = @intCast(yuyv[i + 2]);
        const v: i32 = @as(i32, yuyv[i + 3]) - 128;

        const c0 = y0 - 16;
        const c1 = y1 - 16;
        const r_uv = 409 * v + 128;
        const g_uv = -100 * u - 208 * v + 128;
        const b_uv = 516 * u + 128;

        out_rgb[out_i + 0] = clampU8((298 * c0 + r_uv) >> 8);
        out_rgb[out_i + 1] = clampU8((298 * c0 + g_uv) >> 8);
        out_rgb[out_i + 2] = clampU8((298 * c0 + b_uv) >> 8);

        out_rgb[out_i + 3] = clampU8((298 * c1 + r_uv) >> 8);
        out_rgb[out_i + 4] = clampU8((298 * c1 + g_uv) >> 8);
        out_rgb[out_i + 5] = clampU8((298 * c1 + b_uv) >> 8);
    }
}

inline fn clampU8(val: i32) u8 {
    if (val <= 0) return 0;
    if (val >= 255) return 255;
    return @intCast(val);
}

test "yuyvToRgb24 converts neutral gray and white macro-pixels" {
    // Y=235, U=128, V=128 -> near-white (255, 255, 255)
    const yuyv = [_]u8{ 235, 128, 16, 128 };
    var rgb: [6]u8 = undefined;
    yuyvToRgb24(&yuyv, 2, 1, &rgb);
    try std.testing.expectEqual(@as(u8, 255), rgb[0]);
    try std.testing.expectEqual(@as(u8, 255), rgb[1]);
    try std.testing.expectEqual(@as(u8, 255), rgb[2]);
    // Y=16, U=128, V=128 -> black (0, 0, 0)
    try std.testing.expectEqual(@as(u8, 0), rgb[3]);
    try std.testing.expectEqual(@as(u8, 0), rgb[4]);
    try std.testing.expectEqual(@as(u8, 0), rgb[5]);
}
