const std = @import("std");
const preprocessing = @import("preprocessing.zig");
const postprocessing = @import("postprocessing.zig");

pub const ort = @import("ort_c");

pub const DetectorError = error{
    OrtApiUnavailable,
    OrtStatusError,
    InvalidModelInputs,
    InvalidModelOutputs,
    InvalidInputTensorShape,
    InvalidInputTensorType,
    UnsupportedOutputTensor,
    OutOfMemory,
};

pub const DetectorConfig = struct {
    model_path: []const u8,
    intra_op_threads: u32 = 4,
    inter_op_threads: u32 = 1,
    conf_threshold: f32 = 0.25,
    iou_threshold: f32 = 0.45,
    force_nms: bool = false,
};

/// Encapsulates ONNX Runtime session state and preallocated zero-copy I/O tensors.
/// All large buffers are allocated once in `init()` and reused across inferences.
pub const Detector = struct {
    allocator: std.mem.Allocator,
    api: *const ort.OrtApi,
    ort_version: []const u8,
    env: *ort.OrtEnv,
    session_options: *ort.OrtSessionOptions,
    session: *ort.OrtSession,
    memory_info: *ort.OrtMemoryInfo,
    ort_allocator: *ort.OrtAllocator,

    input_name: [:0]u8,
    output_name: [:0]u8,
    input_shape: [4]i64,
    output_shape: [3]i64,
    output_desc: postprocessing.OutputDescriptor,

    input_tensor_buf: []f32,
    output_tensor_buf: []f32,
    input_ort_value: *ort.OrtValue,
    output_ort_value: *ort.OrtValue,

    scratch: postprocessing.ScratchBuffer,
    detections_buf: []postprocessing.Detection,
    config: DetectorConfig,

    pub fn init(allocator: std.mem.Allocator, config: DetectorConfig) DetectorError!Detector {
        const api_base_raw = ort.OrtGetApiBase();
        if (api_base_raw == null) return DetectorError.OrtApiUnavailable;
        const api_base: *const ort.OrtApiBase = @ptrCast(api_base_raw);

        const api_raw = api_base.GetApi.?(ort.ORT_API_VERSION);
        if (api_raw == null) return DetectorError.OrtApiUnavailable;
        const api: *const ort.OrtApi = @ptrCast(api_raw);
        const ort_version = std.mem.span(api_base.GetVersionString.?());

        var env: ?*ort.OrtEnv = null;
        try checkStatus(api, api.CreateEnv.?(ort.ORT_LOGGING_LEVEL_WARNING, "zig-yolo26n", &env));
        errdefer if (env) |e| api.ReleaseEnv.?(e);

        var session_options: ?*ort.OrtSessionOptions = null;
        try checkStatus(api, api.CreateSessionOptions.?(&session_options));
        errdefer if (session_options) |so| api.ReleaseSessionOptions.?(so);

        // Configure deterministic CPU Execution Provider settings
        try checkStatus(api, api.SetIntraOpNumThreads.?(session_options.?, @intCast(config.intra_op_threads)));
        try checkStatus(api, api.SetInterOpNumThreads.?(session_options.?, @intCast(config.inter_op_threads)));
        try checkStatus(api, api.SetSessionExecutionMode.?(session_options.?, ort.ORT_SEQUENTIAL));
        try checkStatus(api, api.SetSessionGraphOptimizationLevel.?(session_options.?, ort.ORT_ENABLE_ALL));
        try checkStatus(api, api.EnableCpuMemArena.?(session_options.?));
        try checkStatus(api, api.EnableMemPattern.?(session_options.?));

        const model_path_z = allocator.dupeSentinel(u8, config.model_path, 0) catch return DetectorError.OutOfMemory;
        defer allocator.free(model_path_z);

        var session: ?*ort.OrtSession = null;
        try checkStatus(api, api.CreateSession.?(env.?, model_path_z.ptr, session_options.?, &session));
        errdefer if (session) |s| api.ReleaseSession.?(s);

        var ort_allocator: ?*ort.OrtAllocator = null;
        try checkStatus(api, api.GetAllocatorWithDefaultOptions.?(&ort_allocator));

        // Validate input count, name, type, and shape [1, 3, 640, 640]
        var input_count: usize = 0;
        try checkStatus(api, api.SessionGetInputCount.?(session.?, &input_count));
        if (input_count != 1) return DetectorError.InvalidModelInputs;

        var raw_in_name: [*c]u8 = null;
        try checkStatus(api, api.SessionGetInputName.?(session.?, 0, ort_allocator.?, &raw_in_name));
        const input_name = allocator.dupeSentinel(u8, std.mem.span(raw_in_name), 0) catch {
            _ = api.AllocatorFree.?(ort_allocator.?, raw_in_name);
            return DetectorError.OutOfMemory;
        };
        errdefer allocator.free(input_name);
        try checkStatus(api, api.AllocatorFree.?(ort_allocator.?, raw_in_name));

        const input_shape = try validateInputTensor(api, session.?);

        // Validate output count, name, type, and shape
        var output_count: usize = 0;
        try checkStatus(api, api.SessionGetOutputCount.?(session.?, &output_count));
        if (output_count != 1) return DetectorError.InvalidModelOutputs;

        var raw_out_name: [*c]u8 = null;
        try checkStatus(api, api.SessionGetOutputName.?(session.?, 0, ort_allocator.?, &raw_out_name));
        const output_name = allocator.dupeSentinel(u8, std.mem.span(raw_out_name), 0) catch {
            _ = api.AllocatorFree.?(ort_allocator.?, raw_out_name);
            return DetectorError.OutOfMemory;
        };
        errdefer allocator.free(output_name);
        try checkStatus(api, api.AllocatorFree.?(ort_allocator.?, raw_out_name));

        const output_shape = try validateOutputTensor(api, session.?);
        const output_desc = postprocessing.inferOutputDescriptor(output_shape) catch {
            return DetectorError.UnsupportedOutputTensor;
        };

        var memory_info: ?*ort.OrtMemoryInfo = null;
        try checkStatus(api, api.CreateCpuMemoryInfo.?(ort.OrtArenaAllocator, ort.OrtMemTypeDefault, &memory_info));
        errdefer if (memory_info) |mi| api.ReleaseMemoryInfo.?(mi);

        // Preallocate reusable FP32 tensor buffers once at startup
        const in_elems: usize = preprocessing.INPUT_TENSOR_ELEMENTS;
        const input_tensor_buf = allocator.alloc(f32, in_elems) catch return DetectorError.OutOfMemory;
        errdefer allocator.free(input_tensor_buf);
        @memset(input_tensor_buf, 0.0);

        const out_elems: usize = @as(usize, @intCast(output_shape[0])) *
            @as(usize, @intCast(output_shape[1])) *
            @as(usize, @intCast(output_shape[2]));
        const output_tensor_buf = allocator.alloc(f32, out_elems) catch return DetectorError.OutOfMemory;
        errdefer allocator.free(output_tensor_buf);
        @memset(output_tensor_buf, 0.0);

        var input_ort_value: ?*ort.OrtValue = null;
        try checkStatus(api, api.CreateTensorWithDataAsOrtValue.?(
            memory_info.?,
            input_tensor_buf.ptr,
            input_tensor_buf.len * @sizeOf(f32),
            &input_shape,
            input_shape.len,
            ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_ort_value,
        ));
        errdefer if (input_ort_value) |v| api.ReleaseValue.?(v);

        var output_ort_value: ?*ort.OrtValue = null;
        try checkStatus(api, api.CreateTensorWithDataAsOrtValue.?(
            memory_info.?,
            output_tensor_buf.ptr,
            output_tensor_buf.len * @sizeOf(f32),
            &output_shape,
            output_shape.len,
            ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &output_ort_value,
        ));
        errdefer if (output_ort_value) |v| api.ReleaseValue.?(v);

        const max_candidates: usize = switch (output_desc.format) {
            .yolo26_end2end => output_desc.dim1,
            .yolo_raw_ch_first => output_desc.dim2,
            .yolo_raw_ch_last => output_desc.dim1,
        };
        var scratch = postprocessing.ScratchBuffer.init(allocator, max_candidates) catch {
            return DetectorError.OutOfMemory;
        };
        errdefer scratch.deinit(allocator);

        const detections_buf = allocator.alloc(postprocessing.Detection, postprocessing.MAX_DETECTIONS) catch {
            return DetectorError.OutOfMemory;
        };

        return .{
            .allocator = allocator,
            .api = api,
            .ort_version = ort_version,
            .env = env.?,
            .session_options = session_options.?,
            .session = session.?,
            .memory_info = memory_info.?,
            .ort_allocator = ort_allocator.?,
            .input_name = input_name,
            .output_name = output_name,
            .input_shape = input_shape,
            .output_shape = output_shape,
            .output_desc = output_desc,
            .input_tensor_buf = input_tensor_buf,
            .output_tensor_buf = output_tensor_buf,
            .input_ort_value = input_ort_value.?,
            .output_ort_value = output_ort_value.?,
            .scratch = scratch,
            .detections_buf = detections_buf,
            .config = config,
        };
    }

    pub fn deinit(self: *Detector) void {
        self.allocator.free(self.detections_buf);
        self.scratch.deinit(self.allocator);
        self.api.ReleaseValue.?(self.output_ort_value);
        self.api.ReleaseValue.?(self.input_ort_value);
        self.allocator.free(self.output_tensor_buf);
        self.allocator.free(self.input_tensor_buf);
        self.api.ReleaseMemoryInfo.?(self.memory_info);
        self.allocator.free(self.output_name);
        self.allocator.free(self.input_name);
        self.api.ReleaseSession.?(self.session);
        self.api.ReleaseSessionOptions.?(self.session_options);
        self.api.ReleaseEnv.?(self.env);
    }

    /// Runs synchronous CPU inference using preallocated zero-copy input and output OrtValues.
    pub fn inferRaw(self: *Detector) DetectorError!void {
        const input_names = [_][*c]const u8{self.input_name.ptr};
        const input_values = [_]?*const ort.OrtValue{self.input_ort_value};
        const output_names = [_][*c]const u8{self.output_name.ptr};
        var output_values = [_]?*ort.OrtValue{self.output_ort_value};

        try checkStatus(self.api, self.api.Run.?(
            self.session,
            null,
            &input_names,
            &input_values,
            1,
            &output_names,
            1,
            &output_values,
        ));
    }

    /// Preprocesses RGB image into the preallocated NCHW input buffer.
    pub fn preprocess(self: *Detector, rgb_data: []const u8, width: u32, height: u32) preprocessing.LetterboxParams {
        return preprocessing.preprocessRgbToNchw(
            rgb_data,
            width,
            height,
            preprocessing.INPUT_WIDTH,
            preprocessing.INPUT_HEIGHT,
            self.input_tensor_buf,
        );
    }

    /// Decodes the preallocated output tensor buffer into `self.detections_buf`.
    pub fn postprocess(self: *Detector, lb: preprocessing.LetterboxParams) []postprocessing.Detection {
        return postprocessing.decode(
            self.output_tensor_buf,
            self.output_desc,
            lb,
            self.config.conf_threshold,
            self.config.iou_threshold,
            self.config.force_nms,
            &self.scratch,
            self.detections_buf,
        );
    }
};

fn checkStatus(api: *const ort.OrtApi, status: ?*ort.OrtStatus) DetectorError!void {
    if (status) |st| {
        const msg = api.GetErrorMessage.?(st);
        std.log.err("ONNX Runtime error: {s}", .{std.mem.span(msg)});
        api.ReleaseStatus.?(st);
        return DetectorError.OrtStatusError;
    }
}

fn validateInputTensor(api: *const ort.OrtApi, session: *ort.OrtSession) DetectorError![4]i64 {
    var type_info: ?*ort.OrtTypeInfo = null;
    try checkStatus(api, api.SessionGetInputTypeInfo.?(session, 0, &type_info));
    defer if (type_info) |ti| api.ReleaseTypeInfo.?(ti);

    var tensor_info: ?*const ort.OrtTensorTypeAndShapeInfo = null;
    try checkStatus(api, api.CastTypeInfoToTensorInfo.?(type_info.?, &tensor_info));
    if (tensor_info == null) return DetectorError.InvalidInputTensorType;

    var elem_type: ort.ONNXTensorElementDataType = ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    try checkStatus(api, api.GetTensorElementType.?(tensor_info.?, &elem_type));
    if (elem_type != ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return DetectorError.InvalidInputTensorType;
    }

    var dims_count: usize = 0;
    try checkStatus(api, api.GetDimensionsCount.?(tensor_info.?, &dims_count));
    if (dims_count != 4) return DetectorError.InvalidInputTensorShape;

    var dims: [4]i64 = undefined;
    try checkStatus(api, api.GetDimensions.?(tensor_info.?, &dims, 4));
    if (dims[0] != 1 or
        dims[1] != preprocessing.INPUT_CHANNELS or
        dims[2] != preprocessing.INPUT_HEIGHT or
        dims[3] != preprocessing.INPUT_WIDTH)
    {
        return DetectorError.InvalidInputTensorShape;
    }
    return dims;
}

fn validateOutputTensor(api: *const ort.OrtApi, session: *ort.OrtSession) DetectorError![3]i64 {
    var type_info: ?*ort.OrtTypeInfo = null;
    try checkStatus(api, api.SessionGetOutputTypeInfo.?(session, 0, &type_info));
    defer if (type_info) |ti| api.ReleaseTypeInfo.?(ti);

    var tensor_info: ?*const ort.OrtTensorTypeAndShapeInfo = null;
    try checkStatus(api, api.CastTypeInfoToTensorInfo.?(type_info.?, &tensor_info));
    if (tensor_info == null) return DetectorError.UnsupportedOutputTensor;

    var elem_type: ort.ONNXTensorElementDataType = ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    try checkStatus(api, api.GetTensorElementType.?(tensor_info.?, &elem_type));
    if (elem_type != ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return DetectorError.UnsupportedOutputTensor;
    }

    var dims_count: usize = 0;
    try checkStatus(api, api.GetDimensionsCount.?(tensor_info.?, &dims_count));
    if (dims_count != 3) return DetectorError.UnsupportedOutputTensor;

    var dims: [3]i64 = undefined;
    try checkStatus(api, api.GetDimensions.?(tensor_info.?, &dims, 3));
    return dims;
}
