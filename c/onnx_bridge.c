#include "onnxruntime_c_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const OrtApi* api;
    OrtEnv* env;
    OrtSessionOptions* session_options;
    OrtSession* session;
    OrtMemoryInfo* memory_info;
    OrtAllocator* ort_allocator;
    OrtValue* input_value;
    OrtValue* output_value;
    char* input_name;
    char* output_name;
    char ort_version[64];
} OrtBridgeContext;

static void set_err(char* err_buf, size_t err_cap, const char* msg) {
    if (err_buf != NULL && err_cap > 0) {
        snprintf(err_buf, err_cap, "%s", msg != NULL ? msg : "Unknown error");
    }
}

static int check_status(const OrtApi* api, OrtStatus* status, char* err_buf, size_t err_cap) {
    if (status == NULL) return 0;
    const char* msg = api->GetErrorMessage(status);
    set_err(err_buf, err_cap, msg);
    api->ReleaseStatus(status);
    return -1;
}

void ort_bridge_destroy(void* ctx_ptr) {
    OrtBridgeContext* ctx = (OrtBridgeContext*)ctx_ptr;
    if (ctx == NULL) return;
    if (ctx->api != NULL) {
        if (ctx->output_value != NULL) ctx->api->ReleaseValue(ctx->output_value);
        if (ctx->input_value != NULL) ctx->api->ReleaseValue(ctx->input_value);
        if (ctx->memory_info != NULL) ctx->api->ReleaseMemoryInfo(ctx->memory_info);
        if (ctx->session != NULL) ctx->api->ReleaseSession(ctx->session);
        if (ctx->session_options != NULL) ctx->api->ReleaseSessionOptions(ctx->session_options);
        if (ctx->env != NULL) ctx->api->ReleaseEnv(ctx->env);
    }
    free(ctx->input_name);
    free(ctx->output_name);
    free(ctx);
}

void* ort_bridge_create(
    const char* model_path,
    int32_t intra_op_threads,
    int32_t inter_op_threads,
    char* out_ort_version,
    size_t ort_version_cap,
    int64_t* out_input_dims,
    int64_t* out_output_dims,
    char* err_buf,
    size_t err_cap) {
    OrtBridgeContext* ctx = (OrtBridgeContext*)calloc(1, sizeof(OrtBridgeContext));
    if (ctx == NULL) {
        set_err(err_buf, err_cap, "Out of memory allocating OrtBridgeContext");
        return NULL;
    }

    const OrtApiBase* api_base = OrtGetApiBase();
    if (api_base == NULL) {
        set_err(err_buf, err_cap, "OrtGetApiBase returned NULL");
        free(ctx);
        return NULL;
    }
    ctx->api = api_base->GetApi(ORT_API_VERSION);
    if (ctx->api == NULL) {
        set_err(err_buf, err_cap, "Requested ORT_API_VERSION is unavailable");
        free(ctx);
        return NULL;
    }

    const char* ver = api_base->GetVersionString();
    snprintf(ctx->ort_version, sizeof(ctx->ort_version), "%s", ver != NULL ? ver : "unknown");
    if (out_ort_version != NULL && ort_version_cap > 0) {
        snprintf(out_ort_version, ort_version_cap, "%s", ctx->ort_version);
    }

    if (check_status(ctx->api, ctx->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "fortran-yolo26n", &ctx->env), err_buf, err_cap) != 0) {
        ort_bridge_destroy(ctx);
        return NULL;
    }
    if (check_status(ctx->api, ctx->api->CreateSessionOptions(&ctx->session_options), err_buf, err_cap) != 0) {
        ort_bridge_destroy(ctx);
        return NULL;
    }
    if (check_status(ctx->api, ctx->api->SetIntraOpNumThreads(ctx->session_options, intra_op_threads), err_buf, err_cap) != 0 ||
        check_status(ctx->api, ctx->api->SetInterOpNumThreads(ctx->session_options, inter_op_threads), err_buf, err_cap) != 0 ||
        check_status(ctx->api, ctx->api->SetSessionExecutionMode(ctx->session_options, ORT_SEQUENTIAL), err_buf, err_cap) != 0 ||
        check_status(ctx->api, ctx->api->SetSessionGraphOptimizationLevel(ctx->session_options, ORT_ENABLE_ALL), err_buf, err_cap) != 0 ||
        check_status(ctx->api, ctx->api->EnableCpuMemArena(ctx->session_options), err_buf, err_cap) != 0 ||
        check_status(ctx->api, ctx->api->EnableMemPattern(ctx->session_options), err_buf, err_cap) != 0) {
        ort_bridge_destroy(ctx);
        return NULL;
    }

    if (check_status(ctx->api, ctx->api->CreateSession(ctx->env, model_path, ctx->session_options, &ctx->session), err_buf, err_cap) != 0) {
        ort_bridge_destroy(ctx);
        return NULL;
    }
    if (check_status(ctx->api, ctx->api->GetAllocatorWithDefaultOptions(&ctx->ort_allocator), err_buf, err_cap) != 0) {
        ort_bridge_destroy(ctx);
        return NULL;
    }

    size_t in_count = 0;
    size_t out_count = 0;
    if (check_status(ctx->api, ctx->api->SessionGetInputCount(ctx->session, &in_count), err_buf, err_cap) != 0 ||
        check_status(ctx->api, ctx->api->SessionGetOutputCount(ctx->session, &out_count), err_buf, err_cap) != 0) {
        ort_bridge_destroy(ctx);
        return NULL;
    }
    if (in_count != 1 || out_count != 1) {
        set_err(err_buf, err_cap, "Model must have exactly 1 input and 1 output");
        ort_bridge_destroy(ctx);
        return NULL;
    }

    char* raw_in_name = NULL;
    char* raw_out_name = NULL;
    if (check_status(ctx->api, ctx->api->SessionGetInputName(ctx->session, 0, ctx->ort_allocator, &raw_in_name), err_buf, err_cap) != 0) {
        ort_bridge_destroy(ctx);
        return NULL;
    }
    ctx->input_name = strdup(raw_in_name);
    if (check_status(ctx->api, ctx->api->AllocatorFree(ctx->ort_allocator, raw_in_name), err_buf, err_cap) != 0) {
        ort_bridge_destroy(ctx);
        return NULL;
    }

    if (check_status(ctx->api, ctx->api->SessionGetOutputName(ctx->session, 0, ctx->ort_allocator, &raw_out_name), err_buf, err_cap) != 0) {
        ort_bridge_destroy(ctx);
        return NULL;
    }
    ctx->output_name = strdup(raw_out_name);
    if (check_status(ctx->api, ctx->api->AllocatorFree(ctx->ort_allocator, raw_out_name), err_buf, err_cap) != 0) {
        ort_bridge_destroy(ctx);
        return NULL;
    }

    // Validate input tensor type (FP32) and rank (4)
    OrtTypeInfo* in_type_info = NULL;
    if (check_status(ctx->api, ctx->api->SessionGetInputTypeInfo(ctx->session, 0, &in_type_info), err_buf, err_cap) != 0) {
        ort_bridge_destroy(ctx);
        return NULL;
    }
    const OrtTensorTypeAndShapeInfo* in_tensor_info = NULL;
    if (check_status(ctx->api, ctx->api->CastTypeInfoToTensorInfo(in_type_info, &in_tensor_info), err_buf, err_cap) != 0 || in_tensor_info == NULL) {
        ctx->api->ReleaseTypeInfo(in_type_info);
        set_err(err_buf, err_cap, "Input is not a tensor");
        ort_bridge_destroy(ctx);
        return NULL;
    }
    ONNXTensorElementDataType in_elem = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    size_t in_rank = 0;
    if (check_status(ctx->api, ctx->api->GetTensorElementType(in_tensor_info, &in_elem), err_buf, err_cap) != 0 ||
        check_status(ctx->api, ctx->api->GetDimensionsCount(in_tensor_info, &in_rank), err_buf, err_cap) != 0 ||
        in_elem != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || in_rank != 4 ||
        check_status(ctx->api, ctx->api->GetDimensions(in_tensor_info, out_input_dims, 4), err_buf, err_cap) != 0) {
        ctx->api->ReleaseTypeInfo(in_type_info);
        set_err(err_buf, err_cap, "Invalid input tensor type or rank (expected 4D FP32)");
        ort_bridge_destroy(ctx);
        return NULL;
    }
    ctx->api->ReleaseTypeInfo(in_type_info);

    // Validate output tensor type (FP32) and rank (3)
    OrtTypeInfo* out_type_info = NULL;
    if (check_status(ctx->api, ctx->api->SessionGetOutputTypeInfo(ctx->session, 0, &out_type_info), err_buf, err_cap) != 0) {
        ort_bridge_destroy(ctx);
        return NULL;
    }
    const OrtTensorTypeAndShapeInfo* out_tensor_info = NULL;
    if (check_status(ctx->api, ctx->api->CastTypeInfoToTensorInfo(out_type_info, &out_tensor_info), err_buf, err_cap) != 0 || out_tensor_info == NULL) {
        ctx->api->ReleaseTypeInfo(out_type_info);
        set_err(err_buf, err_cap, "Output is not a tensor");
        ort_bridge_destroy(ctx);
        return NULL;
    }
    ONNXTensorElementDataType out_elem = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    size_t out_rank = 0;
    if (check_status(ctx->api, ctx->api->GetTensorElementType(out_tensor_info, &out_elem), err_buf, err_cap) != 0 ||
        check_status(ctx->api, ctx->api->GetDimensionsCount(out_tensor_info, &out_rank), err_buf, err_cap) != 0 ||
        out_elem != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || out_rank != 3 ||
        check_status(ctx->api, ctx->api->GetDimensions(out_tensor_info, out_output_dims, 3), err_buf, err_cap) != 0) {
        ctx->api->ReleaseTypeInfo(out_type_info);
        set_err(err_buf, err_cap, "Invalid output tensor type or rank (expected 3D FP32)");
        ort_bridge_destroy(ctx);
        return NULL;
    }
    ctx->api->ReleaseTypeInfo(out_type_info);

    if (check_status(ctx->api, ctx->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ctx->memory_info), err_buf, err_cap) != 0) {
        ort_bridge_destroy(ctx);
        return NULL;
    }

    return ctx;
}

int ort_bridge_bind_tensors(
    void* ctx_ptr,
    float* input_buf,
    size_t input_bytes,
    const int64_t* input_dims,
    float* output_buf,
    size_t output_bytes,
    const int64_t* output_dims,
    char* err_buf,
    size_t err_cap) {
    OrtBridgeContext* ctx = (OrtBridgeContext*)ctx_ptr;
    if (ctx == NULL) {
        set_err(err_buf, err_cap, "Null OrtBridgeContext");
        return -1;
    }

    if (ctx->input_value != NULL) {
        ctx->api->ReleaseValue(ctx->input_value);
        ctx->input_value = NULL;
    }
    if (ctx->output_value != NULL) {
        ctx->api->ReleaseValue(ctx->output_value);
        ctx->output_value = NULL;
    }

    if (check_status(ctx->api, ctx->api->CreateTensorWithDataAsOrtValue(
            ctx->memory_info, input_buf, input_bytes, input_dims, 4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ctx->input_value), err_buf, err_cap) != 0) {
        return -1;
    }

    if (check_status(ctx->api, ctx->api->CreateTensorWithDataAsOrtValue(
            ctx->memory_info, output_buf, output_bytes, output_dims, 3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ctx->output_value), err_buf, err_cap) != 0) {
        return -1;
    }

    return 0;
}

int ort_bridge_run(void* ctx_ptr, char* err_buf, size_t err_cap) {
    OrtBridgeContext* ctx = (OrtBridgeContext*)ctx_ptr;
    if (ctx == NULL || ctx->input_value == NULL || ctx->output_value == NULL) {
        set_err(err_buf, err_cap, "ORT session or tensors not initialized");
        return -1;
    }
    const char* input_names[1] = {ctx->input_name};
    const OrtValue* input_values[1] = {ctx->input_value};
    const char* output_names[1] = {ctx->output_name};
    OrtValue* output_values[1] = {ctx->output_value};

    return check_status(ctx->api, ctx->api->Run(
        ctx->session, NULL,
        input_names, input_values, 1,
        output_names, 1, output_values), err_buf, err_cap);
}
