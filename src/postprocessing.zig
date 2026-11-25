const std = @import("std");
const preprocessing = @import("preprocessing.zig");

pub const NUM_COCO_CLASSES: usize = 80;
pub const MAX_DETECTIONS: usize = 300;

pub const COCO_CLASSES = [NUM_COCO_CLASSES][]const u8{
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

pub fn className(class_id: u32) []const u8 {
    if (class_id < NUM_COCO_CLASSES) return COCO_CLASSES[class_id];
    return "unknown";
}

pub const Detection = struct {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    confidence: f32,
    class_id: u32,
};

/// Supported YOLO ONNX output layouts:
/// - `yolo26_end2end`: [1, N, 6] with [x1, y1, x2, y2, score, class_id] (NMS-free one-to-one head)
/// - `yolo_raw_ch_first`: [1, 4 + num_classes, num_anchors] (one-to-many head, requires NMS)
/// - `yolo_raw_ch_last`: [1, num_anchors, 4 + num_classes] (transposed raw head, requires NMS)
pub const OutputFormat = enum {
    yolo26_end2end,
    yolo_raw_ch_first,
    yolo_raw_ch_last,

    pub fn requiresNms(self: OutputFormat) bool {
        return switch (self) {
            .yolo26_end2end => false,
            .yolo_raw_ch_first, .yolo_raw_ch_last => true,
        };
    }
};

pub const OutputDescriptor = struct {
    format: OutputFormat,
    dim1: usize,
    dim2: usize,
    num_classes: usize,
};

pub const PostprocessError = error{
    UnsupportedOutputShape,
};

/// Inspects validated [1, D1, D2] ONNX output tensor dimensions to select the
/// appropriate YOLO26 decoder without assuming YOLOv8's [1, 84, 8400] layout.
pub fn inferOutputDescriptor(shape: [3]i64) PostprocessError!OutputDescriptor {
    if (shape[0] != 1 or shape[1] <= 0 or shape[2] <= 0) {
        return PostprocessError.UnsupportedOutputShape;
    }
    const d1: usize = @intCast(shape[1]);
    const d2: usize = @intCast(shape[2]);

    // Canonical YOLO26 end2end format: [1, max_det, 6]
    if (d2 == 6) {
        return .{
            .format = .yolo26_end2end,
            .dim1 = d1,
            .dim2 = d2,
            .num_classes = NUM_COCO_CLASSES,
        };
    }
    // Raw channel-first format: [1, 4 + nc, num_anchors], e.g. [1, 84, 8400]
    if (d1 == 4 + NUM_COCO_CLASSES and d2 > 6) {
        return .{
            .format = .yolo_raw_ch_first,
            .dim1 = d1,
            .dim2 = d2,
            .num_classes = NUM_COCO_CLASSES,
        };
    }
    // Raw channel-last format: [1, num_anchors, 4 + nc], e.g. [1, 8400, 84]
    if (d2 == 4 + NUM_COCO_CLASSES and d1 > 6) {
        return .{
            .format = .yolo_raw_ch_last,
            .dim1 = d1,
            .dim2 = d2,
            .num_classes = NUM_COCO_CLASSES,
        };
    }
    return PostprocessError.UnsupportedOutputShape;
}

/// Reusable scratch buffer for NMS when decoding raw one-to-many outputs.
pub const ScratchBuffer = struct {
    candidates: []Detection,
    suppressed: []bool,

    pub fn init(allocator: std.mem.Allocator, max_candidates: usize) !ScratchBuffer {
        const candidates = try allocator.alloc(Detection, max_candidates);
        errdefer allocator.free(candidates);
        const suppressed = try allocator.alloc(bool, max_candidates);
        return .{
            .candidates = candidates,
            .suppressed = suppressed,
        };
    }

    pub fn deinit(self: *ScratchBuffer, allocator: std.mem.Allocator) void {
        allocator.free(self.candidates);
        allocator.free(self.suppressed);
    }
};

/// Decodes raw FP32 ONNX output tensor into `out_detections` using zero heap allocations.
/// Returns the slice of valid detections sorted by descending confidence.
pub fn decode(
    raw_output: []const f32,
    desc: OutputDescriptor,
    lb: preprocessing.LetterboxParams,
    conf_threshold: f32,
    iou_threshold: f32,
    force_nms: bool,
    scratch: *ScratchBuffer,
    out_detections: []Detection,
) []Detection {
    const count = switch (desc.format) {
        .yolo26_end2end => decodeEnd2End(
            raw_output,
            desc.dim1,
            lb,
            conf_threshold,
            if (force_nms) scratch.candidates else out_detections,
        ),
        .yolo_raw_ch_first => decodeRawChannelFirst(
            raw_output,
            desc.dim2,
            desc.num_classes,
            lb,
            conf_threshold,
            scratch.candidates,
        ),
        .yolo_raw_ch_last => decodeRawChannelLast(
            raw_output,
            desc.dim1,
            desc.num_classes,
            lb,
            conf_threshold,
            scratch.candidates,
        ),
    };

    const run_nms = desc.format.requiresNms() or force_nms;
    if (!run_nms) {
        const res = out_detections[0..count];
        sortByConfidenceDesc(res);
        return res;
    }

    const cand = scratch.candidates[0..count];
    sortByConfidenceDesc(cand);
    return applyClassAwareNms(cand, scratch.suppressed[0..count], iou_threshold, out_detections);
}

fn decodeEnd2End(
    raw_output: []const f32,
    num_rows: usize,
    lb: preprocessing.LetterboxParams,
    conf_threshold: f32,
    out: []Detection,
) usize {
    var n: usize = 0;
    const inv_scale = 1.0 / lb.scale;
    const fpad_x: f32 = @floatFromInt(lb.pad_x);
    const fpad_y: f32 = @floatFromInt(lb.pad_y);
    const max_w: f32 = @floatFromInt(lb.orig_w);
    const max_h: f32 = @floatFromInt(lb.orig_h);

    for (0..num_rows) |i| {
        const base = i * 6;
        const score = raw_output[base + 4];
        if (!std.math.isFinite(score) or score < conf_threshold) continue;

        const cls_f = raw_output[base + 5];
        if (!std.math.isFinite(cls_f) or cls_f < 0.0) continue;
        const cls_id: u32 = @intFromFloat(@round(cls_f));
        if (cls_id >= NUM_COCO_CLASSES) continue;

        const x1 = std.math.clamp((raw_output[base + 0] - fpad_x) * inv_scale, 0.0, max_w);
        const y1 = std.math.clamp((raw_output[base + 1] - fpad_y) * inv_scale, 0.0, max_h);
        const x2 = std.math.clamp((raw_output[base + 2] - fpad_x) * inv_scale, 0.0, max_w);
        const y2 = std.math.clamp((raw_output[base + 3] - fpad_y) * inv_scale, 0.0, max_h);
        if (x2 <= x1 or y2 <= y1) continue;

        if (n < out.len) {
            out[n] = .{
                .x1 = x1,
                .y1 = y1,
                .x2 = x2,
                .y2 = y2,
                .confidence = std.math.clamp(score, 0.0, 1.0),
                .class_id = cls_id,
            };
            n += 1;
        }
    }
    return n;
}

fn decodeRawChannelFirst(
    raw_output: []const f32,
    num_anchors: usize,
    num_classes: usize,
    lb: preprocessing.LetterboxParams,
    conf_threshold: f32,
    out: []Detection,
) usize {
    var n: usize = 0;
    const inv_scale = 1.0 / lb.scale;
    const fpad_x: f32 = @floatFromInt(lb.pad_x);
    const fpad_y: f32 = @floatFromInt(lb.pad_y);
    const max_w: f32 = @floatFromInt(lb.orig_w);
    const max_h: f32 = @floatFromInt(lb.orig_h);

    for (0..num_anchors) |a| {
        var best_score: f32 = -1.0;
        var best_cls: u32 = 0;
        for (0..num_classes) |c| {
            const s = raw_output[(4 + c) * num_anchors + a];
            if (s > best_score) {
                best_score = s;
                best_cls = @intCast(c);
            }
        }
        if (!std.math.isFinite(best_score) or best_score < conf_threshold) continue;

        const cx = raw_output[0 * num_anchors + a];
        const cy = raw_output[1 * num_anchors + a];
        const w = raw_output[2 * num_anchors + a];
        const h = raw_output[3 * num_anchors + a];
        const half_w = w * 0.5;
        const half_h = h * 0.5;

        const x1 = std.math.clamp(((cx - half_w) - fpad_x) * inv_scale, 0.0, max_w);
        const y1 = std.math.clamp(((cy - half_h) - fpad_y) * inv_scale, 0.0, max_h);
        const x2 = std.math.clamp(((cx + half_w) - fpad_x) * inv_scale, 0.0, max_w);
        const y2 = std.math.clamp(((cy + half_h) - fpad_y) * inv_scale, 0.0, max_h);
        if (x2 <= x1 or y2 <= y1) continue;

        if (n < out.len) {
            out[n] = .{
                .x1 = x1,
                .y1 = y1,
                .x2 = x2,
                .y2 = y2,
                .confidence = std.math.clamp(best_score, 0.0, 1.0),
                .class_id = best_cls,
            };
            n += 1;
        }
    }
    return n;
}

fn decodeRawChannelLast(
    raw_output: []const f32,
    num_anchors: usize,
    num_classes: usize,
    lb: preprocessing.LetterboxParams,
    conf_threshold: f32,
    out: []Detection,
) usize {
    var n: usize = 0;
    const stride = 4 + num_classes;
    const inv_scale = 1.0 / lb.scale;
    const fpad_x: f32 = @floatFromInt(lb.pad_x);
    const fpad_y: f32 = @floatFromInt(lb.pad_y);
    const max_w: f32 = @floatFromInt(lb.orig_w);
    const max_h: f32 = @floatFromInt(lb.orig_h);

    for (0..num_anchors) |a| {
        const base = a * stride;
        var best_score: f32 = -1.0;
        var best_cls: u32 = 0;
        for (0..num_classes) |c| {
            const s = raw_output[base + 4 + c];
            if (s > best_score) {
                best_score = s;
                best_cls = @intCast(c);
            }
        }
        if (!std.math.isFinite(best_score) or best_score < conf_threshold) continue;

        const cx = raw_output[base + 0];
        const cy = raw_output[base + 1];
        const w = raw_output[base + 2];
        const h = raw_output[base + 3];
        const half_w = w * 0.5;
        const half_h = h * 0.5;

        const x1 = std.math.clamp(((cx - half_w) - fpad_x) * inv_scale, 0.0, max_w);
        const y1 = std.math.clamp(((cy - half_h) - fpad_y) * inv_scale, 0.0, max_h);
        const x2 = std.math.clamp(((cx + half_w) - fpad_x) * inv_scale, 0.0, max_w);
        const y2 = std.math.clamp(((cy + half_h) - fpad_y) * inv_scale, 0.0, max_h);
        if (x2 <= x1 or y2 <= y1) continue;

        if (n < out.len) {
            out[n] = .{
                .x1 = x1,
                .y1 = y1,
                .x2 = x2,
                .y2 = y2,
                .confidence = std.math.clamp(best_score, 0.0, 1.0),
                .class_id = best_cls,
            };
            n += 1;
        }
    }
    return n;
}

fn sortByConfidenceDesc(items: []Detection) void {
    std.sort.pdq(Detection, items, {}, struct {
        fn lessThan(_: void, a: Detection, b: Detection) bool {
            return a.confidence > b.confidence;
        }
    }.lessThan);
}

fn computeIou(a: Detection, b: Detection) f32 {
    const ix1 = @max(a.x1, b.x1);
    const iy1 = @max(a.y1, b.y1);
    const ix2 = @min(a.x2, b.x2);
    const iy2 = @min(a.y2, b.y2);
    const iw = @max(0.0, ix2 - ix1);
    const ih = @max(0.0, iy2 - iy1);
    const inter = iw * ih;
    if (inter <= 0.0) return 0.0;

    const area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    const union_area = area_a + area_b - inter;
    if (union_area <= 0.0) return 0.0;
    return inter / union_area;
}

fn applyClassAwareNms(
    sorted_candidates: []const Detection,
    suppressed: []bool,
    iou_threshold: f32,
    out_detections: []Detection,
) []Detection {
    @memset(suppressed, false);
    var out_count: usize = 0;

    for (sorted_candidates, 0..) |cand, i| {
        if (suppressed[i]) continue;
        if (out_count >= out_detections.len) break;

        out_detections[out_count] = cand;
        out_count += 1;

        for (i + 1..sorted_candidates.len) |j| {
            if (suppressed[j]) continue;
            if (sorted_candidates[j].class_id != cand.class_id) continue;
            if (computeIou(cand, sorted_candidates[j]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return out_detections[0..out_count];
}

test "inferOutputDescriptor distinguishes YOLO26 end2end vs YOLOv8 raw" {
    const d_e2e = try inferOutputDescriptor(.{ 1, 300, 6 });
    try std.testing.expectEqual(OutputFormat.yolo26_end2end, d_e2e.format);
    try std.testing.expect(!d_e2e.format.requiresNms());

    const d_raw = try inferOutputDescriptor(.{ 1, 84, 8400 });
    try std.testing.expectEqual(OutputFormat.yolo_raw_ch_first, d_raw.format);
    try std.testing.expect(d_raw.format.requiresNms());
}

test "decodeEnd2End unscales letterbox coordinates and clamps to image bounds" {
    const lb = preprocessing.computeLetterbox(810, 1080, 640, 640);
    // Row 0: valid bus box in 640x640 letterbox coords; Row 1: below threshold
    const raw = [_]f32{
        80.0, 135.0, 560.0, 445.0, 0.92, 5.0,
        10.0, 10.0,  50.0,  50.0,  0.10, 0.0,
    };
    const desc = try inferOutputDescriptor(.{ 1, 2, 6 });
    var scratch = try ScratchBuffer.init(std.testing.allocator, 16);
    defer scratch.deinit(std.testing.allocator);
    var out: [16]Detection = undefined;

    const dets = decode(&raw, desc, lb, 0.25, 0.45, false, &scratch, &out);
    try std.testing.expectEqual(@as(usize, 1), dets.len);
    try std.testing.expectEqual(@as(u32, 5), dets[0].class_id);
    try std.testing.expect(dets[0].x1 >= 0.0 and dets[0].x2 <= 810.0);
    try std.testing.expect(dets[0].y1 >= 0.0 and dets[0].y2 <= 1080.0);
}
