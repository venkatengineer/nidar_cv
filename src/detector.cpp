#include "detector.hpp"

#include <algorithm>
#include <stdexcept>

namespace yolo {

Detector::Detector(const DetectorConfig& config)
    : config_(config),
      scratch_(0),
      detections_buf_(MAX_DETECTIONS) {
    try {
        const OrtApiBase* api_base = OrtGetApiBase();
        if (api_base == nullptr) {
            throw std::runtime_error("OrtGetApiBase returned null");
        }
        api_ = api_base->GetApi(ORT_API_VERSION);
        if (api_ == nullptr) {
            throw std::runtime_error("ONNX Runtime C API version unsupported");
        }
        ort_version_ = api_base->GetVersionString();

        checkStatus(api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "cpp-yolo26n", &env_));
        checkStatus(api_->CreateSessionOptions(&session_options_));

        checkStatus(api_->SetIntraOpNumThreads(session_options_, static_cast<int>(config_.intra_op_threads)));
        checkStatus(api_->SetInterOpNumThreads(session_options_, static_cast<int>(config_.inter_op_threads)));
        checkStatus(api_->SetSessionExecutionMode(session_options_, ORT_SEQUENTIAL));
        checkStatus(api_->SetSessionGraphOptimizationLevel(session_options_, ORT_ENABLE_ALL));
        checkStatus(api_->EnableCpuMemArena(session_options_));
        checkStatus(api_->EnableMemPattern(session_options_));

        checkStatus(api_->CreateSession(env_, config_.model_path.c_str(), session_options_, &session_));
        checkStatus(api_->GetAllocatorWithDefaultOptions(&ort_allocator_));

        size_t input_count = 0;
        checkStatus(api_->SessionGetInputCount(session_, &input_count));
        if (input_count != 1) {
            throw std::runtime_error("Expected exactly 1 model input");
        }

        char* raw_in_name = nullptr;
        checkStatus(api_->SessionGetInputName(session_, 0, ort_allocator_, &raw_in_name));
        input_name_ = raw_in_name;
        checkStatus(api_->AllocatorFree(ort_allocator_, raw_in_name));

        input_shape_ = validateInputTensor();

        size_t output_count = 0;
        checkStatus(api_->SessionGetOutputCount(session_, &output_count));
        if (output_count != 1) {
            throw std::runtime_error("Expected exactly 1 model output");
        }

        char* raw_out_name = nullptr;
        checkStatus(api_->SessionGetOutputName(session_, 0, ort_allocator_, &raw_out_name));
        output_name_ = raw_out_name;
        checkStatus(api_->AllocatorFree(ort_allocator_, raw_out_name));

        output_shape_ = validateOutputTensor();
        output_desc_ = inferOutputDescriptor(output_shape_);

        checkStatus(api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_));

        input_tensor_buf_.assign(INPUT_TENSOR_ELEMENTS, 0.0f);
        const size_t out_elems = static_cast<size_t>(output_shape_[0]) *
                                 static_cast<size_t>(output_shape_[1]) *
                                 static_cast<size_t>(output_shape_[2]);
        output_tensor_buf_.assign(out_elems, 0.0f);

        checkStatus(api_->CreateTensorWithDataAsOrtValue(
            memory_info_,
            input_tensor_buf_.data(),
            input_tensor_buf_.size() * sizeof(float),
            input_shape_.data(),
            input_shape_.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_ort_value_));

        checkStatus(api_->CreateTensorWithDataAsOrtValue(
            memory_info_,
            output_tensor_buf_.data(),
            output_tensor_buf_.size() * sizeof(float),
            output_shape_.data(),
            output_shape_.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &output_ort_value_));

        size_t max_candidates = 0;
        switch (output_desc_.format) {
            case OutputFormat::yolo26_end2end:
                max_candidates = output_desc_.dim1;
                break;
            case OutputFormat::yolo_raw_ch_first:
                max_candidates = output_desc_.dim2;
                break;
            case OutputFormat::yolo_raw_ch_last:
                max_candidates = output_desc_.dim1;
                break;
        }
        scratch_ = ScratchBuffer(max_candidates);
    } catch (...) {
        cleanup();
        throw;
    }
}

Detector::~Detector() {
    cleanup();
}

void Detector::cleanup() noexcept {
    if (api_ == nullptr) return;
    if (output_ort_value_ != nullptr) {
        api_->ReleaseValue(output_ort_value_);
        output_ort_value_ = nullptr;
    }
    if (input_ort_value_ != nullptr) {
        api_->ReleaseValue(input_ort_value_);
        input_ort_value_ = nullptr;
    }
    if (memory_info_ != nullptr) {
        api_->ReleaseMemoryInfo(memory_info_);
        memory_info_ = nullptr;
    }
    if (session_ != nullptr) {
        api_->ReleaseSession(session_);
        session_ = nullptr;
    }
    if (session_options_ != nullptr) {
        api_->ReleaseSessionOptions(session_options_);
        session_options_ = nullptr;
    }
    if (env_ != nullptr) {
        api_->ReleaseEnv(env_);
        env_ = nullptr;
    }
}

void Detector::checkStatus(OrtStatus* status) const {
    if (status != nullptr) {
        const char* msg = api_->GetErrorMessage(status);
        std::string err_msg = msg != nullptr ? msg : "Unknown ONNX Runtime error";
        api_->ReleaseStatus(status);
        throw std::runtime_error("ONNX Runtime error: " + err_msg);
    }
}

std::array<int64_t, 4> Detector::validateInputTensor() const {
    OrtTypeInfo* type_info = nullptr;
    checkStatus(api_->SessionGetInputTypeInfo(session_, 0, &type_info));
    struct Guard {
        const OrtApi* api;
        OrtTypeInfo* ti;
        ~Guard() { if (ti != nullptr) api->ReleaseTypeInfo(ti); }
    } guard{api_, type_info};

    const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    checkStatus(api_->CastTypeInfoToTensorInfo(type_info, &tensor_info));
    if (tensor_info == nullptr) {
        throw std::runtime_error("Model input is not a tensor");
    }

    ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    checkStatus(api_->GetTensorElementType(tensor_info, &elem_type));
    if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("Model input is not FP32");
    }

    size_t dims_count = 0;
    checkStatus(api_->GetDimensionsCount(tensor_info, &dims_count));
    if (dims_count != 4) {
        throw std::runtime_error("Model input rank must be 4");
    }

    std::array<int64_t, 4> dims{};
    checkStatus(api_->GetDimensions(tensor_info, dims.data(), 4));
    if (dims[0] != 1 ||
        dims[1] != static_cast<int64_t>(INPUT_CHANNELS) ||
        dims[2] != static_cast<int64_t>(INPUT_HEIGHT) ||
        dims[3] != static_cast<int64_t>(INPUT_WIDTH)) {
        throw std::runtime_error("Model input shape must be [1, 3, 640, 640]");
    }
    return dims;
}

std::array<int64_t, 3> Detector::validateOutputTensor() const {
    OrtTypeInfo* type_info = nullptr;
    checkStatus(api_->SessionGetOutputTypeInfo(session_, 0, &type_info));
    struct Guard {
        const OrtApi* api;
        OrtTypeInfo* ti;
        ~Guard() { if (ti != nullptr) api->ReleaseTypeInfo(ti); }
    } guard{api_, type_info};

    const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    checkStatus(api_->CastTypeInfoToTensorInfo(type_info, &tensor_info));
    if (tensor_info == nullptr) {
        throw std::runtime_error("Model output is not a tensor");
    }

    ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    checkStatus(api_->GetTensorElementType(tensor_info, &elem_type));
    if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("Model output is not FP32");
    }

    size_t dims_count = 0;
    checkStatus(api_->GetDimensionsCount(tensor_info, &dims_count));
    if (dims_count != 3) {
        throw std::runtime_error("Model output rank must be 3");
    }

    std::array<int64_t, 3> dims{};
    checkStatus(api_->GetDimensions(tensor_info, dims.data(), 3));
    return dims;
}

LetterboxParams Detector::preprocess(const uint8_t* rgb_data, uint32_t width, uint32_t height) {
    return preprocessRgbToNchw(
        rgb_data,
        width,
        height,
        INPUT_WIDTH,
        INPUT_HEIGHT,
        input_tensor_buf_.data(),
        input_tensor_buf_.size());
}

void Detector::inferRaw() {
    const char* input_names[] = {input_name_.c_str()};
    const OrtValue* input_values[] = {input_ort_value_};
    const char* output_names[] = {output_name_.c_str()};
    OrtValue* output_values[] = {output_ort_value_};

    checkStatus(api_->Run(
        session_,
        nullptr,
        input_names,
        input_values,
        1,
        output_names,
        1,
        output_values));
}

size_t Detector::postprocess(const LetterboxParams& lb) {
    return decodeOutput(
        output_tensor_buf_.data(),
        output_desc_,
        lb,
        config_.conf_threshold,
        config_.iou_threshold,
        config_.force_nms,
        scratch_,
        detections_buf_.data(),
        detections_buf_.size());
}

} // namespace yolo
