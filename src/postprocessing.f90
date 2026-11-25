module postprocessing_mod
    use, intrinsic :: iso_c_binding
    use, intrinsic :: iso_fortran_env, only: int32, int64, real32
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use preprocessing_mod, only: letterbox_params_t
    implicit none
    private

    public :: NUM_COCO_CLASSES, MAX_DETECTIONS
    public :: FMT_YOLO26_END2END, FMT_RAW_CH_FIRST, FMT_RAW_CH_LAST
    public :: detection_t, output_descriptor_t, scratch_buffer_t
    public :: coco_class_name, output_format_name, format_requires_nms
    public :: infer_output_descriptor, init_scratch, destroy_scratch, decode_output

    integer(int32), parameter :: NUM_COCO_CLASSES = 80
    integer(int32), parameter :: MAX_DETECTIONS = 300
    integer(int32), parameter :: FMT_YOLO26_END2END = 1
    integer(int32), parameter :: FMT_RAW_CH_FIRST = 2
    integer(int32), parameter :: FMT_RAW_CH_LAST = 3

    type :: detection_t
        real(real32) :: x1 = 0.0_real32
        real(real32) :: y1 = 0.0_real32
        real(real32) :: x2 = 0.0_real32
        real(real32) :: y2 = 0.0_real32
        real(real32) :: confidence = 0.0_real32
        integer(int32) :: class_id = 0
    end type detection_t

    type :: output_descriptor_t
        integer(int32) :: format = FMT_YOLO26_END2END
        integer(int32) :: dim1 = 0
        integer(int32) :: dim2 = 0
        integer(int32) :: num_classes = NUM_COCO_CLASSES
    end type output_descriptor_t

    type :: scratch_buffer_t
        type(detection_t), allocatable :: candidates(:)
        logical, allocatable :: suppressed(:)
    end type scratch_buffer_t

    character(len=16), parameter :: COCO_CLASSES(NUM_COCO_CLASSES) = [ &
        "person          ", "bicycle         ", "car             ", "motorcycle      ", "airplane        ", &
        "bus             ", "train           ", "truck           ", "boat            ", "traffic light   ", &
        "fire hydrant    ", "stop sign       ", "parking meter   ", "bench           ", "bird            ", &
        "cat             ", "dog             ", "horse           ", "sheep           ", "cow             ", &
        "elephant        ", "bear            ", "zebra           ", "giraffe         ", "backpack        ", &
        "umbrella        ", "handbag         ", "tie             ", "suitcase        ", "frisbee         ", &
        "skis            ", "snowboard       ", "sports ball     ", "kite            ", "baseball bat    ", &
        "baseball glove  ", "skateboard      ", "surfboard       ", "tennis racket   ", "bottle          ", &
        "wine glass      ", "cup             ", "fork            ", "knife           ", "spoon           ", &
        "bowl            ", "banana          ", "apple           ", "sandwich        ", "orange          ", &
        "broccoli        ", "carrot          ", "hot dog         ", "pizza           ", "donut           ", &
        "cake            ", "chair           ", "couch           ", "potted plant    ", "bed             ", &
        "dining table    ", "toilet          ", "tv              ", "laptop          ", "mouse           ", &
        "remote          ", "keyboard        ", "cell phone      ", "microwave       ", "oven            ", &
        "toaster         ", "sink            ", "refrigerator    ", "book            ", "clock           ", &
        "vase            ", "scissors        ", "teddy bear      ", "hair drier      ", "toothbrush      " &
    ]

contains

    pure function coco_class_name(class_id) result(name)
        integer(int32), intent(in) :: class_id
        character(len=:), allocatable :: name
        if (class_id >= 0 .and. class_id < NUM_COCO_CLASSES) then
            name = trim(COCO_CLASSES(class_id + 1))
        else
            name = "unknown"
        end if
    end function coco_class_name

    pure function output_format_name(fmt) result(name)
        integer(int32), intent(in) :: fmt
        character(len=:), allocatable :: name
        select case (fmt)
        case (FMT_YOLO26_END2END)
            name = "yolo26_end2end"
        case (FMT_RAW_CH_FIRST)
            name = "yolo_raw_ch_first"
        case (FMT_RAW_CH_LAST)
            name = "yolo_raw_ch_last"
        case default
            name = "unknown"
        end select
    end function output_format_name

    pure function format_requires_nms(fmt) result(req)
        integer(int32), intent(in) :: fmt
        logical :: req
        req = (fmt /= FMT_YOLO26_END2END)
    end function format_requires_nms

    function infer_output_descriptor(shape, desc) result(ok)
        integer(c_int64_t), intent(in) :: shape(3)
        type(output_descriptor_t), intent(out) :: desc
        logical :: ok
        integer(int32) :: d1, d2

        ok = .false.
        if (shape(1) /= 1 .or. shape(2) <= 0 .or. shape(3) <= 0) return
        d1 = int(shape(2), int32)
        d2 = int(shape(3), int32)

        if (d2 == 6) then
            desc = output_descriptor_t(FMT_YOLO26_END2END, d1, d2, NUM_COCO_CLASSES)
            ok = .true.
        else if (d1 == 4 + NUM_COCO_CLASSES .and. d2 > 6) then
            desc = output_descriptor_t(FMT_RAW_CH_FIRST, d1, d2, NUM_COCO_CLASSES)
            ok = .true.
        else if (d2 == 4 + NUM_COCO_CLASSES .and. d1 > 6) then
            desc = output_descriptor_t(FMT_RAW_CH_LAST, d1, d2, NUM_COCO_CLASSES)
            ok = .true.
        end if
    end function infer_output_descriptor

    subroutine init_scratch(scratch, max_candidates)
        type(scratch_buffer_t), intent(out) :: scratch
        integer(int32), intent(in) :: max_candidates
        allocate(scratch%candidates(max_candidates))
        allocate(scratch%suppressed(max_candidates))
        scratch%suppressed = .false.
    end subroutine init_scratch

    subroutine destroy_scratch(scratch)
        type(scratch_buffer_t), intent(inout) :: scratch
        if (allocated(scratch%candidates)) deallocate(scratch%candidates)
        if (allocated(scratch%suppressed)) deallocate(scratch%suppressed)
    end subroutine destroy_scratch

    subroutine sort_detections_desc(arr, n)
        type(detection_t), intent(inout) :: arr(:)
        integer(int32), intent(in) :: n
        integer(int32) :: i, j
        type(detection_t) :: key

        ! N <= 300 in end2end; insertion sort is branch-light and zero-allocation.
        do i = 2, n
            key = arr(i)
            j = i - 1
            do while (j >= 1)
                if (arr(j)%confidence >= key%confidence) exit
                arr(j + 1) = arr(j)
                j = j - 1
            end do
            arr(j + 1) = key
        end do
    end subroutine sort_detections_desc

    pure function compute_iou(a, b) result(iou)
        type(detection_t), intent(in) :: a, b
        real(real32) :: iou, ix1, iy1, ix2, iy2, iw, ih, inter, area_a, area_b, union_a

        ix1 = max(a%x1, b%x1)
        iy1 = max(a%y1, b%y1)
        ix2 = min(a%x2, b%x2)
        iy2 = min(a%y2, b%y2)
        iw = max(0.0_real32, ix2 - ix1)
        ih = max(0.0_real32, iy2 - iy1)
        inter = iw * ih
        if (inter <= 0.0_real32) then
            iou = 0.0_real32
            return
        end if
        area_a = (a%x2 - a%x1) * (a%y2 - a%y1)
        area_b = (b%x2 - b%x1) * (b%y2 - b%y1)
        union_a = area_a + area_b - inter
        if (union_a <= 0.0_real32) then
            iou = 0.0_real32
        else
            iou = inter / union_a
        end if
    end function compute_iou

    ! `raw_output` has Fortran shape (desc%dim2, desc%dim1), matching C [1, dim1, dim2].
    function decode_output(raw_output, desc, lb, conf_threshold, iou_threshold, force_nms, scratch, out_dets) result(count)
        real(c_float), intent(in) :: raw_output(:, :)
        type(output_descriptor_t), intent(in) :: desc
        type(letterbox_params_t), intent(in) :: lb
        real(real32), intent(in) :: conf_threshold, iou_threshold
        logical, intent(in) :: force_nms
        type(scratch_buffer_t), intent(inout) :: scratch
        type(detection_t), intent(out) :: out_dets(:)
        integer(int32) :: count

        integer(int32) :: cand_count, i, j, out_cap, out_idx
        logical :: run_nms
        real(real32) :: inv_scale, fpad_x, fpad_y, max_w, max_h
        real(real32) :: score, cls_f, x1, y1, x2, y2, cx, cy, w, h, hw, hh, best_score, s
        integer(int32) :: cls_id, best_cls, a, c

        inv_scale = 1.0_real32 / lb%scale
        fpad_x = real(lb%pad_x, real32)
        fpad_y = real(lb%pad_y, real32)
        max_w = real(lb%orig_w, real32)
        max_h = real(lb%orig_h, real32)
        run_nms = format_requires_nms(desc%format) .or. force_nms
        cand_count = 0

        select case (desc%format)
        case (FMT_YOLO26_END2END)
            out_cap = size(out_dets)
            if (run_nms) out_cap = size(scratch%candidates)
            do i = 1, desc%dim1
                score = raw_output(5, i)
                if (.not. ieee_is_finite(score) .or. score < conf_threshold) cycle
                cls_f = raw_output(6, i)
                if (.not. ieee_is_finite(cls_f) .or. cls_f < 0.0_real32) cycle
                cls_id = nint(cls_f, int32)
                if (cls_id >= NUM_COCO_CLASSES) cycle

                x1 = min(max((raw_output(1, i) - fpad_x) * inv_scale, 0.0_real32), max_w)
                y1 = min(max((raw_output(2, i) - fpad_y) * inv_scale, 0.0_real32), max_h)
                x2 = min(max((raw_output(3, i) - fpad_x) * inv_scale, 0.0_real32), max_w)
                y2 = min(max((raw_output(4, i) - fpad_y) * inv_scale, 0.0_real32), max_h)
                if (x2 <= x1 .or. y2 <= y1) cycle

                if (cand_count < out_cap) then
                    cand_count = cand_count + 1
                    score = min(max(score, 0.0_real32), 1.0_real32)
                    if (run_nms) then
                        scratch%candidates(cand_count) = detection_t(x1, y1, x2, y2, score, cls_id)
                    else
                        out_dets(cand_count) = detection_t(x1, y1, x2, y2, score, cls_id)
                    end if
                end if
            end do

        case (FMT_RAW_CH_FIRST)
            ! C [1, 84, 8400] -> Fortran (8400, 84)
            out_cap = size(scratch%candidates)
            do a = 1, desc%dim2
                best_score = -1.0_real32
                best_cls = 0
                do c = 1, desc%num_classes
                    s = raw_output(a, 4 + c)
                    if (s > best_score) then
                        best_score = s
                        best_cls = c - 1
                    end if
                end do
                if (.not. ieee_is_finite(best_score) .or. best_score < conf_threshold) cycle
                cx = raw_output(a, 1)
                cy = raw_output(a, 2)
                w = raw_output(a, 3)
                h = raw_output(a, 4)
                hw = w * 0.5_real32
                hh = h * 0.5_real32
                x1 = min(max(((cx - hw) - fpad_x) * inv_scale, 0.0_real32), max_w)
                y1 = min(max(((cy - hh) - fpad_y) * inv_scale, 0.0_real32), max_h)
                x2 = min(max(((cx + hw) - fpad_x) * inv_scale, 0.0_real32), max_w)
                y2 = min(max(((cy + hh) - fpad_y) * inv_scale, 0.0_real32), max_h)
                if (x2 <= x1 .or. y2 <= y1) cycle
                if (cand_count < out_cap) then
                    cand_count = cand_count + 1
                    best_score = min(max(best_score, 0.0_real32), 1.0_real32)
                    scratch%candidates(cand_count) = detection_t(x1, y1, x2, y2, best_score, best_cls)
                end if
            end do

        case (FMT_RAW_CH_LAST)
            ! C [1, 8400, 84] -> Fortran (84, 8400)
            out_cap = size(scratch%candidates)
            do a = 1, desc%dim1
                best_score = -1.0_real32
                best_cls = 0
                do c = 1, desc%num_classes
                    s = raw_output(4 + c, a)
                    if (s > best_score) then
                        best_score = s
                        best_cls = c - 1
                    end if
                end do
                if (.not. ieee_is_finite(best_score) .or. best_score < conf_threshold) cycle
                cx = raw_output(1, a)
                cy = raw_output(2, a)
                w = raw_output(3, a)
                h = raw_output(4, a)
                hw = w * 0.5_real32
                hh = h * 0.5_real32
                x1 = min(max(((cx - hw) - fpad_x) * inv_scale, 0.0_real32), max_w)
                y1 = min(max(((cy - hh) - fpad_y) * inv_scale, 0.0_real32), max_h)
                x2 = min(max(((cx + hw) - fpad_x) * inv_scale, 0.0_real32), max_w)
                y2 = min(max(((cy + hh) - fpad_y) * inv_scale, 0.0_real32), max_h)
                if (x2 <= x1 .or. y2 <= y1) cycle
                if (cand_count < out_cap) then
                    cand_count = cand_count + 1
                    best_score = min(max(best_score, 0.0_real32), 1.0_real32)
                    scratch%candidates(cand_count) = detection_t(x1, y1, x2, y2, best_score, best_cls)
                end if
            end do
        end select

        if (.not. run_nms) then
            call sort_detections_desc(out_dets, cand_count)
            count = cand_count
            return
        end if

        call sort_detections_desc(scratch%candidates, cand_count)
        scratch%suppressed(1:cand_count) = .false.
        out_idx = 0
        out_cap = size(out_dets)

        do i = 1, cand_count
            if (scratch%suppressed(i)) cycle
            if (out_idx >= out_cap) exit
            out_idx = out_idx + 1
            out_dets(out_idx) = scratch%candidates(i)
            do j = i + 1, cand_count
                if (scratch%suppressed(j)) cycle
                if (scratch%candidates(j)%class_id /= scratch%candidates(i)%class_id) cycle
                if (compute_iou(scratch%candidates(i), scratch%candidates(j)) > iou_threshold) then
                    scratch%suppressed(j) = .true.
                end if
            end do
        end do
        count = out_idx
    end function decode_output

end module postprocessing_mod
