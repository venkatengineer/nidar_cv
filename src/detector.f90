module detector_mod
    use, intrinsic :: iso_c_binding
    use, intrinsic :: iso_fortran_env, only: int32, int64, real32
    use onnx_runtime_mod
    use preprocessing_mod
    use postprocessing_mod
    implicit none
    private

    public :: detector_config_t, detector_t
    public :: detector_init, detector_preprocess, detector_infer_raw
    public :: detector_postprocess, detector_destroy

    type :: detector_config_t
        character(len=512) :: model_path = "models/yolov26n.onnx"
        integer(int32) :: intra_op_threads = 4
        integer(int32) :: inter_op_threads = 1
        real(real32) :: conf_threshold = 0.25_real32
        real(real32) :: iou_threshold = 0.45_real32
        logical :: force_nms = .false.
    end type detector_config_t

    ! `input_tensor` and `output_tensor` carry the TARGET attribute so ONNX Runtime
    ! can wrap them zero-copy via `c_loc(...)`. They are allocated once in
    ! `detector_init` and released only AFTER `ort_destroy_session` in `detector_destroy`.
    type :: detector_t
        type(detector_config_t) :: config
        type(ort_session_t) :: session
        type(output_descriptor_t) :: output_desc
        real(c_float), allocatable :: input_tensor(:, :, :)
        real(c_float), allocatable :: output_tensor(:, :)
        type(scratch_buffer_t) :: scratch
        type(detection_t), allocatable :: detections(:)
    end type detector_t

contains

    function detector_init(det, cfg, err_msg) result(ok)
        type(detector_t), target, intent(out) :: det
        type(detector_config_t), intent(in) :: cfg
        character(len=*), intent(out) :: err_msg
        logical :: ok

        integer(int32) :: max_cand

        det%config = cfg
        err_msg = ""

        if (.not. ort_init_session(det%session, cfg%model_path, &
                int(cfg%intra_op_threads, c_int32_t), int(cfg%inter_op_threads, c_int32_t), err_msg)) then
            ok = .false.
            return
        end if

        if (det%session%input_dims(1) /= 1 .or. &
            det%session%input_dims(2) /= INPUT_CHANNELS .or. &
            det%session%input_dims(3) /= INPUT_HEIGHT .or. &
            det%session%input_dims(4) /= INPUT_WIDTH) then
            err_msg = "Invalid input tensor shape (expected [1, 3, 640, 640])"
            call ort_destroy_session(det%session)
            ok = .false.
            return
        end if

        if (.not. infer_output_descriptor(det%session%output_dims, det%output_desc)) then
            err_msg = "Unsupported output tensor shape"
            call ort_destroy_session(det%session)
            ok = .false.
            return
        end if

        ! Fortran column-major (W, H, C) matches C [1, C, H, W];
        ! Fortran (dim2, dim1) matches C [1, dim1, dim2].
        allocate(det%input_tensor(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS))
        allocate(det%output_tensor(det%output_desc%dim2, det%output_desc%dim1))
        det%input_tensor = 0.0_c_float
        det%output_tensor = 0.0_c_float

        if (.not. ort_bind_tensors(det%session, det%input_tensor, det%output_tensor, err_msg)) then
            call detector_destroy(det)
            ok = .false.
            return
        end if

        select case (det%output_desc%format)
        case (FMT_YOLO26_END2END)
            max_cand = det%output_desc%dim1
        case (FMT_RAW_CH_FIRST)
            max_cand = det%output_desc%dim2
        case (FMT_RAW_CH_LAST)
            max_cand = det%output_desc%dim1
        case default
            max_cand = MAX_DETECTIONS
        end select

        call init_scratch(det%scratch, max_cand)
        allocate(det%detections(MAX_DETECTIONS))
        ok = .true.
    end function detector_init

    function detector_preprocess(det, rgb_data, width, height) result(lb)
        type(detector_t), intent(inout) :: det
        integer(c_int8_t), intent(in) :: rgb_data(:, :, :)
        integer(int32), intent(in) :: width, height
        type(letterbox_params_t) :: lb

        lb = preprocess_rgb_to_nchw(rgb_data, width, height, INPUT_WIDTH, INPUT_HEIGHT, det%input_tensor)
    end function detector_preprocess

    function detector_infer_raw(det, err_msg) result(ok)
        type(detector_t), intent(inout) :: det
        character(len=*), intent(out) :: err_msg
        logical :: ok

        ok = ort_run_inference(det%session, err_msg)
    end function detector_infer_raw

    function detector_postprocess(det, lb) result(count)
        type(detector_t), intent(inout) :: det
        type(letterbox_params_t), intent(in) :: lb
        integer(int32) :: count

        count = decode_output( &
            det%output_tensor, &
            det%output_desc, &
            lb, &
            det%config%conf_threshold, &
            det%config%iou_threshold, &
            det%config%force_nms, &
            det%scratch, &
            det%detections)
    end function detector_postprocess

    subroutine detector_destroy(det)
        type(detector_t), intent(inout) :: det

        ! Release ORT handles BEFORE freeing TARGET Fortran buffers
        call ort_destroy_session(det%session)
        call destroy_scratch(det%scratch)
        if (allocated(det%detections)) deallocate(det%detections)
        if (allocated(det%output_tensor)) deallocate(det%output_tensor)
        if (allocated(det%input_tensor)) deallocate(det%input_tensor)
    end subroutine detector_destroy

end module detector_mod
