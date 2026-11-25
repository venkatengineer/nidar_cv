const std = @import("std");
const postprocessing = @import("postprocessing.zig");

const c = @import("stb_c");

pub const ImageError = error{
    LoadFailed,
    SaveFailed,
    OutOfMemory,
};

/// Owned RGB24 image buffer (3 bytes per pixel, row-major).
pub const Image = struct {
    width: u32,
    height: u32,
    channels: u32,
    data: []u8,
    backing: Backing,

    const Backing = union(enum) {
        stb: void,
        zig_allocator: std.mem.Allocator,
    };

    /// Loads any JPEG/PNG/BMP image from disk and normalizes it to 3-channel RGB24.
    /// The returned buffer is managed by stb_image and freed on `deinit()`.
    pub fn load(allocator: std.mem.Allocator, path: []const u8) ImageError!Image {
        const path_z = allocator.dupeSentinel(u8, path, 0) catch return ImageError.OutOfMemory;
        defer allocator.free(path_z);

        var w: c_int = 0;
        var h: c_int = 0;
        var orig_channels: c_int = 0;
        const ptr = c.stbi_load(path_z.ptr, &w, &h, &orig_channels, 3);
        if (ptr == null or w <= 0 or h <= 0) {
            return ImageError.LoadFailed;
        }

        const uw: u32 = @intCast(w);
        const uh: u32 = @intCast(h);
        const byte_len: usize = @as(usize, uw) * @as(usize, uh) * 3;
        return .{
            .width = uw,
            .height = uh,
            .channels = 3,
            .data = ptr[0..byte_len],
            .backing = .stb,
        };
    }

    /// Creates an explicit Zig-allocator-owned copy from a borrowed RGB24 slice.
    pub fn cloneFromRgb(allocator: std.mem.Allocator, rgb: []const u8, width: u32, height: u32) ImageError!Image {
        const copy = allocator.dupe(u8, rgb) catch return ImageError.OutOfMemory;
        return .{
            .width = width,
            .height = height,
            .channels = 3,
            .data = copy,
            .backing = .{ .zig_allocator = allocator },
        };
    }

    /// Creates an explicit Zig-allocator-owned copy for annotation/visualization.
    pub fn clone(self: *const Image, allocator: std.mem.Allocator) ImageError!Image {
        const copy = allocator.dupe(u8, self.data) catch return ImageError.OutOfMemory;
        return .{
            .width = self.width,
            .height = self.height,
            .channels = self.channels,
            .data = copy,
            .backing = .{ .zig_allocator = allocator },
        };
    }

    pub fn deinit(self: *Image) void {
        switch (self.backing) {
            .stb => c.stbi_image_free(self.data.ptr),
            .zig_allocator => |alloc| alloc.free(self.data),
        }
        self.data = &.{};
    }

    pub fn savePng(self: *const Image, allocator: std.mem.Allocator, path: []const u8) ImageError!void {
        const path_z = allocator.dupeSentinel(u8, path, 0) catch return ImageError.OutOfMemory;
        defer allocator.free(path_z);

        const stride: c_int = @intCast(self.width * self.channels);
        const ok = c.stbi_write_png(
            path_z.ptr,
            @intCast(self.width),
            @intCast(self.height),
            @intCast(self.channels),
            self.data.ptr,
            stride,
        );
        if (ok == 0) return ImageError.SaveFailed;
    }

    /// Draws bounding boxes and class/confidence labels directly on the RGB buffer.
    pub fn drawDetections(self: *Image, detections: []const postprocessing.Detection) void {
        var label_buf: [64]u8 = undefined;
        for (detections) |det| {
            const color = classColor(det.class_id);
            const x1 = clampCoord(det.x1, self.width -| 1);
            const y1 = clampCoord(det.y1, self.height -| 1);
            const x2 = clampCoord(det.x2, self.width -| 1);
            const y2 = clampCoord(det.y2, self.height -| 1);
            if (x2 <= x1 or y2 <= y1) continue;

            self.drawRectOutline(x1, y1, x2, y2, 3, color);

            const class_name = postprocessing.className(det.class_id);
            const conf_pct: u32 = @intFromFloat(@round(@min(@max(det.confidence, 0.0), 1.0) * 100.0));
            const label = std.fmt.bufPrint(&label_buf, "{s} {d}%", .{ class_name, conf_pct }) catch class_name;

            const scale: u32 = 2;
            const text_w: u32 = @as(u32, @intCast(label.len)) * 6 * scale + 6;
            const text_h: u32 = 7 * scale + 6;
            const label_y: u32 = if (y1 >= text_h) y1 - text_h else y1;
            const label_x2: u32 = @min(x1 + text_w, self.width -| 1);
            const label_y2: u32 = @min(label_y + text_h, self.height -| 1);

            self.fillRect(x1, label_y, label_x2, label_y2, color);
            const text_color = [3]u8{ 255, 255, 255 };
            self.drawText(x1 + 3, label_y + 3, label, scale, text_color);
        }
    }

    fn setPixel(self: *Image, x: u32, y: u32, color: [3]u8) void {
        if (x >= self.width or y >= self.height) return;
        const idx = (@as(usize, y) * @as(usize, self.width) + @as(usize, x)) * 3;
        self.data[idx] = color[0];
        self.data[idx + 1] = color[1];
        self.data[idx + 2] = color[2];
    }

    fn fillRect(self: *Image, x1: u32, y1: u32, x2: u32, y2: u32, color: [3]u8) void {
        var y = y1;
        while (y <= y2) : (y += 1) {
            var x = x1;
            while (x <= x2) : (x += 1) {
                self.setPixel(x, y, color);
            }
        }
    }

    fn drawRectOutline(self: *Image, x1: u32, y1: u32, x2: u32, y2: u32, thickness: u32, color: [3]u8) void {
        var t: u32 = 0;
        while (t < thickness) : (t += 1) {
            const tx1 = if (x1 + t <= x2) x1 + t else x2;
            const ty1 = if (y1 + t <= y2) y1 + t else y2;
            const tx2 = if (x2 >= x1 + t) x2 - t else x1;
            const ty2 = if (y2 >= y1 + t) y2 - t else y1;

            var x = tx1;
            while (x <= tx2) : (x += 1) {
                self.setPixel(x, ty1, color);
                self.setPixel(x, ty2, color);
            }
            var y = ty1;
            while (y <= ty2) : (y += 1) {
                self.setPixel(tx1, y, color);
                self.setPixel(tx2, y, color);
            }
        }
    }

    fn drawText(self: *Image, x: u32, y: u32, text: []const u8, scale: u32, color: [3]u8) void {
        var cx = x;
        for (text) |ch| {
            const glyph = asciiGlyph5x7(ch);
            for (0..5) |col| {
                const bits = glyph[col];
                for (0..7) |row| {
                    if ((bits & (@as(u8, 1) << @intCast(row))) != 0) {
                        var sy: u32 = 0;
                        while (sy < scale) : (sy += 1) {
                            var sx: u32 = 0;
                            while (sx < scale) : (sx += 1) {
                                self.setPixel(
                                    cx + @as(u32, @intCast(col)) * scale + sx,
                                    y + @as(u32, @intCast(row)) * scale + sy,
                                    color,
                                );
                            }
                        }
                    }
                }
            }
            cx += 6 * scale;
        }
    }
};

fn clampCoord(val: f32, max_val: u32) u32 {
    if (val <= 0.0) return 0;
    const fmax: f32 = @floatFromInt(max_val);
    if (val >= fmax) return max_val;
    return @intFromFloat(val);
}

fn classColor(class_id: u32) [3]u8 {
    const palette = [_][3]u8{
        .{ 220, 50, 47 },
        .{ 38, 139, 210 },
        .{ 133, 153, 0 },
        .{ 211, 54, 130 },
        .{ 42, 161, 152 },
        .{ 203, 75, 22 },
        .{ 108, 113, 196 },
        .{ 181, 137, 0 },
    };
    return palette[class_id % palette.len];
}

/// Minimal 5x7 column-major bitmap font for printable ASCII (uppercase/lowercase/digits/%/space).
fn asciiGlyph5x7(ch_in: u8) [5]u8 {
    const ch = std.ascii.toUpper(ch_in);
    return switch (ch) {
        'A' => .{ 0x7E, 0x11, 0x11, 0x11, 0x7E },
        'B' => .{ 0x7F, 0x49, 0x49, 0x49, 0x36 },
        'C' => .{ 0x3E, 0x41, 0x41, 0x41, 0x22 },
        'D' => .{ 0x7F, 0x41, 0x41, 0x22, 0x1C },
        'E' => .{ 0x7F, 0x49, 0x49, 0x49, 0x41 },
        'F' => .{ 0x7F, 0x09, 0x09, 0x09, 0x01 },
        'G' => .{ 0x3E, 0x41, 0x49, 0x49, 0x7A },
        'H' => .{ 0x7F, 0x08, 0x08, 0x08, 0x7F },
        'I' => .{ 0x00, 0x41, 0x7F, 0x41, 0x00 },
        'J' => .{ 0x20, 0x40, 0x41, 0x3F, 0x01 },
        'K' => .{ 0x7F, 0x08, 0x14, 0x22, 0x41 },
        'L' => .{ 0x7F, 0x40, 0x40, 0x40, 0x40 },
        'M' => .{ 0x7F, 0x02, 0x0C, 0x02, 0x7F },
        'N' => .{ 0x7F, 0x04, 0x08, 0x10, 0x7F },
        'O' => .{ 0x3E, 0x41, 0x41, 0x41, 0x3E },
        'P' => .{ 0x7F, 0x09, 0x09, 0x09, 0x06 },
        'Q' => .{ 0x3E, 0x41, 0x51, 0x21, 0x5E },
        'R' => .{ 0x7F, 0x09, 0x19, 0x29, 0x46 },
        'S' => .{ 0x46, 0x49, 0x49, 0x49, 0x31 },
        'T' => .{ 0x01, 0x01, 0x7F, 0x01, 0x01 },
        'U' => .{ 0x3F, 0x40, 0x40, 0x40, 0x3F },
        'V' => .{ 0x1F, 0x20, 0x40, 0x20, 0x1F },
        'W' => .{ 0x3F, 0x40, 0x38, 0x40, 0x3F },
        'X' => .{ 0x63, 0x14, 0x08, 0x14, 0x63 },
        'Y' => .{ 0x07, 0x08, 0x70, 0x08, 0x07 },
        'Z' => .{ 0x61, 0x51, 0x49, 0x45, 0x43 },
        '0' => .{ 0x3E, 0x51, 0x49, 0x45, 0x3E },
        '1' => .{ 0x00, 0x42, 0x7F, 0x40, 0x00 },
        '2' => .{ 0x42, 0x61, 0x51, 0x49, 0x46 },
        '3' => .{ 0x21, 0x41, 0x45, 0x4B, 0x31 },
        '4' => .{ 0x18, 0x14, 0x12, 0x7F, 0x10 },
        '5' => .{ 0x27, 0x45, 0x45, 0x45, 0x39 },
        '6' => .{ 0x3C, 0x4A, 0x49, 0x49, 0x30 },
        '7' => .{ 0x01, 0x71, 0x09, 0x05, 0x03 },
        '8' => .{ 0x36, 0x49, 0x49, 0x49, 0x36 },
        '9' => .{ 0x06, 0x49, 0x49, 0x29, 0x1E },
        '%' => .{ 0x23, 0x13, 0x08, 0x64, 0x62 },
        '-' => .{ 0x08, 0x08, 0x08, 0x08, 0x08 },
        '_' => .{ 0x40, 0x40, 0x40, 0x40, 0x40 },
        '.' => .{ 0x00, 0x60, 0x60, 0x00, 0x00 },
        ' ' => .{ 0x00, 0x00, 0x00, 0x00, 0x00 },
        else => .{ 0x00, 0x00, 0x00, 0x00, 0x00 },
    };
}
