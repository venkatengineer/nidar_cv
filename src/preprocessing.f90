module preprocessing_mod
    use, intrinsic :: iso_c_binding
    use, intrinsic :: iso_fortran_env, only: int32, real32
    implicit none
    private

    public :: INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS, INPUT_TENSOR_ELEMENTS
    public :: PAD_VALUE_U8, PAD_VALUE_F32
    public :: letterbox_params_t
    public :: compute_letterbox
    public :: preprocess_rgb_to_nchw
    public :: preprocess_rgb_to_nchw_omp
    public :: kernel_rgb_normalize
    public :: kernel_bilinear_resize_rgb
    public :: kernel_letterbox_pad
    public :: kernel_hwc_to_chw

    integer(int32), parameter :: INPUT_WIDTH = 640
    integer(int32), parameter :: INPUT_HEIGHT = 640
    integer(int32), parameter :: INPUT_CHANNELS = 3
    integer(int32), parameter :: INPUT_TENSOR_ELEMENTS = INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH
    integer(int32), parameter :: PAD_VALUE_U8 = 114
    real(real32), parameter :: PAD_VALUE_F32 = real(PAD_VALUE_U8, real32) / 255.0_real32

    type :: letterbox_params_t
        integer(int32) :: orig_w = 0
        integer(int32) :: orig_h = 0
        integer(int32) :: dst_w = 0
        integer(int32) :: dst_h = 0
        integer(int32) :: resized_w = 0
        integer(int32) :: resized_h = 0
        integer(int32) :: pad_x = 0
        integer(int32) :: pad_y = 0
        real(real32) :: scale = 1.0_real32
    end type letterbox_params_t

contains

    pure function compute_letterbox(orig_w, orig_h, dst_w, dst_h) result(p)
        integer(int32), intent(in) :: orig_w, orig_h, dst_w, dst_h
        type(letterbox_params_t) :: p
        real(real32) :: fw, fh, fdw, fdh, dw, dh

        fw = real(orig_w, real32)
        fh = real(orig_h, real32)
        fdw = real(dst_w, real32)
        fdh = real(dst_h, real32)

        p%orig_w = orig_w
        p%orig_h = orig_h
        p%dst_w = dst_w
        p%dst_h = dst_h
        p%scale = min(fdw / fw, fdh / fh)
        p%resized_w = max(1_int32, nint(fw * p%scale, int32))
        p%resized_h = max(1_int32, nint(fh * p%scale, int32))

        dw = real(dst_w - p%resized_w, real32) * 0.5_real32
        dh = real(dst_h - p%resized_h, real32) * 0.5_real32
        p%pad_x = nint(max(0.0_real32, dw - 0.1_real32), int32)
        p%pad_y = nint(max(0.0_real32, dh - 0.1_real32), int32)
    end function compute_letterbox

    ! Fused deterministic preprocessing pipeline matching Zig and C++:
    ! Input `rgb_data(3, orig_w, orig_h)` maps directly to C row-major RGB24.
    ! Output `out_tensor(dst_w, dst_h, 3)` maps directly to C NCHW [1, 3, dst_h, dst_w].
    function preprocess_rgb_to_nchw(rgb_data, orig_w, orig_h, dst_w, dst_h, out_tensor) result(params)
        integer(c_int8_t), intent(in) :: rgb_data(:, :, :)
        integer(int32), intent(in) :: orig_w, orig_h, dst_w, dst_h
        real(c_float), intent(out) :: out_tensor(:, :, :)
        type(letterbox_params_t) :: params

        integer(int32) :: dx, dy, x0, x1, y0, y1
        integer(int32) :: dst_x, dst_y
        real(real32) :: inv_scale_x, inv_scale_y, max_sx, max_sy
        real(real32) :: sx_f, sy_f, wx, wx_inv, wy, wy_inv
        real(real32) :: w00, w01, w10, w11
        real(real32), parameter :: inv_255 = 1.0_real32 / 255.0_real32
        real(real32) :: p00_r, p01_r, p10_r, p11_r
        real(real32) :: p00_g, p01_g, p10_g, p11_g
        real(real32) :: p00_b, p01_b, p10_b, p11_b

        params = compute_letterbox(orig_w, orig_h, dst_w, dst_h)
        out_tensor(:, :, :) = PAD_VALUE_F32

        inv_scale_x = real(orig_w, real32) / real(params%resized_w, real32)
        inv_scale_y = real(orig_h, real32) / real(params%resized_h, real32)
        max_sx = real(orig_w - 1, real32)
        max_sy = real(orig_h - 1, real32)

        do dy = 1, params%resized_h
            sy_f = min(max(((real(dy - 1, real32) + 0.5_real32) * inv_scale_y) - 0.5_real32, 0.0_real32), max_sy)
            y0 = int(floor(sy_f), int32) + 1
            y1 = min(y0 + 1, orig_h)
            wy = sy_f - real(y0 - 1, real32)
            wy_inv = 1.0_real32 - wy
            dst_y = params%pad_y + dy

            do dx = 1, params%resized_w
                sx_f = min(max(((real(dx - 1, real32) + 0.5_real32) * inv_scale_x) - 0.5_real32, 0.0_real32), max_sx)
                x0 = int(floor(sx_f), int32) + 1
                x1 = min(x0 + 1, orig_w)
                wx = sx_f - real(x0 - 1, real32)
                wx_inv = 1.0_real32 - wx

                w00 = wy_inv * wx_inv * inv_255
                w01 = wy_inv * wx * inv_255
                w10 = wy * wx_inv * inv_255
                w11 = wy * wx * inv_255

                p00_r = real(iand(int(rgb_data(1, x0, y0), int32), 255), real32)
                p01_r = real(iand(int(rgb_data(1, x1, y0), int32), 255), real32)
                p10_r = real(iand(int(rgb_data(1, x0, y1), int32), 255), real32)
                p11_r = real(iand(int(rgb_data(1, x1, y1), int32), 255), real32)

                p00_g = real(iand(int(rgb_data(2, x0, y0), int32), 255), real32)
                p01_g = real(iand(int(rgb_data(2, x1, y0), int32), 255), real32)
                p10_g = real(iand(int(rgb_data(2, x0, y1), int32), 255), real32)
                p11_g = real(iand(int(rgb_data(2, x1, y1), int32), 255), real32)

                p00_b = real(iand(int(rgb_data(3, x0, y0), int32), 255), real32)
                p01_b = real(iand(int(rgb_data(3, x1, y0), int32), 255), real32)
                p10_b = real(iand(int(rgb_data(3, x0, y1), int32), 255), real32)
                p11_b = real(iand(int(rgb_data(3, x1, y1), int32), 255), real32)

                dst_x = params%pad_x + dx
                out_tensor(dst_x, dst_y, 1) = p00_r * w00 + p01_r * w01 + p10_r * w10 + p11_r * w11
                out_tensor(dst_x, dst_y, 2) = p00_g * w00 + p01_g * w01 + p10_g * w10 + p11_g * w11
                out_tensor(dst_x, dst_y, 3) = p00_b * w00 + p01_b * w01 + p10_b * w10 + p11_b * w11
            end do
        end do
    end function preprocess_rgb_to_nchw

    ! OpenMP parallel variant of the exact same preprocessing kernel for kernel-bench comparison.
    function preprocess_rgb_to_nchw_omp(rgb_data, orig_w, orig_h, dst_w, dst_h, out_tensor) result(params)
        integer(c_int8_t), intent(in) :: rgb_data(:, :, :)
        integer(int32), intent(in) :: orig_w, orig_h, dst_w, dst_h
        real(c_float), intent(out) :: out_tensor(:, :, :)
        type(letterbox_params_t) :: params

        integer(int32) :: dx, dy, x0, x1, y0, y1, dst_x, dst_y
        real(real32) :: inv_scale_x, inv_scale_y, max_sx, max_sy
        real(real32) :: sx_f, sy_f, wx, wx_inv, wy, wy_inv
        real(real32) :: w00, w01, w10, w11
        real(real32), parameter :: inv_255 = 1.0_real32 / 255.0_real32

        params = compute_letterbox(orig_w, orig_h, dst_w, dst_h)
        out_tensor(:, :, :) = PAD_VALUE_F32

        inv_scale_x = real(orig_w, real32) / real(params%resized_w, real32)
        inv_scale_y = real(orig_h, real32) / real(params%resized_h, real32)
        max_sx = real(orig_w - 1, real32)
        max_sy = real(orig_h - 1, real32)

        !$omp parallel do default(none) &
        !$omp shared(rgb_data, out_tensor, params, orig_w, orig_h, inv_scale_x, inv_scale_y, max_sx, max_sy) &
        !$omp private(dy, dx, sy_f, y0, y1, wy, wy_inv, dst_y, sx_f, x0, x1, wx, wx_inv, w00, w01, w10, w11, dst_x) &
        !$omp schedule(static)
        do dy = 1, params%resized_h
            sy_f = min(max(((real(dy - 1, real32) + 0.5_real32) * inv_scale_y) - 0.5_real32, 0.0_real32), max_sy)
            y0 = int(floor(sy_f), int32) + 1
            y1 = min(y0 + 1, orig_h)
            wy = sy_f - real(y0 - 1, real32)
            wy_inv = 1.0_real32 - wy
            dst_y = params%pad_y + dy

            do dx = 1, params%resized_w
                sx_f = min(max(((real(dx - 1, real32) + 0.5_real32) * inv_scale_x) - 0.5_real32, 0.0_real32), max_sx)
                x0 = int(floor(sx_f), int32) + 1
                x1 = min(x0 + 1, orig_w)
                wx = sx_f - real(x0 - 1, real32)
                wx_inv = 1.0_real32 - wx

                w00 = wy_inv * wx_inv * inv_255
                w01 = wy_inv * wx * inv_255
                w10 = wy * wx_inv * inv_255
                w11 = wy * wx * inv_255

                dst_x = params%pad_x + dx
                out_tensor(dst_x, dst_y, 1) = &
                    real(iand(int(rgb_data(1, x0, y0), int32), 255), real32) * w00 + &
                    real(iand(int(rgb_data(1, x1, y0), int32), 255), real32) * w01 + &
                    real(iand(int(rgb_data(1, x0, y1), int32), 255), real32) * w10 + &
                    real(iand(int(rgb_data(1, x1, y1), int32), 255), real32) * w11
                out_tensor(dst_x, dst_y, 2) = &
                    real(iand(int(rgb_data(2, x0, y0), int32), 255), real32) * w00 + &
                    real(iand(int(rgb_data(2, x1, y0), int32), 255), real32) * w01 + &
                    real(iand(int(rgb_data(2, x0, y1), int32), 255), real32) * w10 + &
                    real(iand(int(rgb_data(2, x1, y1), int32), 255), real32) * w11
                out_tensor(dst_x, dst_y, 3) = &
                    real(iand(int(rgb_data(3, x0, y0), int32), 255), real32) * w00 + &
                    real(iand(int(rgb_data(3, x1, y0), int32), 255), real32) * w01 + &
                    real(iand(int(rgb_data(3, x0, y1), int32), 255), real32) * w10 + &
                    real(iand(int(rgb_data(3, x1, y1), int32), 255), real32) * w11
            end do
        end do
        !$omp end parallel do
    end function preprocess_rgb_to_nchw_omp

    ! Standalone numerical kernels for Section 10 kernel benchmark:

    subroutine kernel_rgb_normalize(rgb_u8, w, h, rgb_f32)
        integer(c_int8_t), intent(in) :: rgb_u8(:, :, :)
        integer(int32), intent(in) :: w, h
        real(c_float), intent(out) :: rgb_f32(:, :, :)
        integer(int32) :: x, y, c
        real(real32), parameter :: inv_255 = 1.0_real32 / 255.0_real32

        do y = 1, h
            do x = 1, w
                do c = 1, 3
                    rgb_f32(c, x, y) = real(iand(int(rgb_u8(c, x, y), int32), 255), real32) * inv_255
                end do
            end do
        end do
    end subroutine kernel_rgb_normalize

    subroutine kernel_bilinear_resize_rgb(src_f32, src_w, src_h, dst_w, dst_h, dst_f32)
        real(c_float), intent(in) :: src_f32(:, :, :)
        integer(int32), intent(in) :: src_w, src_h, dst_w, dst_h
        real(c_float), intent(out) :: dst_f32(:, :, :)
        integer(int32) :: dx, dy, x0, x1, y0, y1, c
        real(real32) :: inv_sx, inv_sy, max_sx, max_sy, sx_f, sy_f, wx, wx_inv, wy, wy_inv
        real(real32) :: w00, w01, w10, w11

        inv_sx = real(src_w, real32) / real(dst_w, real32)
        inv_sy = real(src_h, real32) / real(dst_h, real32)
        max_sx = real(src_w - 1, real32)
        max_sy = real(src_h - 1, real32)

        do dy = 1, dst_h
            sy_f = min(max(((real(dy - 1, real32) + 0.5_real32) * inv_sy) - 0.5_real32, 0.0_real32), max_sy)
            y0 = int(floor(sy_f), int32) + 1
            y1 = min(y0 + 1, src_h)
            wy = sy_f - real(y0 - 1, real32)
            wy_inv = 1.0_real32 - wy

            do dx = 1, dst_w
                sx_f = min(max(((real(dx - 1, real32) + 0.5_real32) * inv_sx) - 0.5_real32, 0.0_real32), max_sx)
                x0 = int(floor(sx_f), int32) + 1
                x1 = min(x0 + 1, src_w)
                wx = sx_f - real(x0 - 1, real32)
                wx_inv = 1.0_real32 - wx

                w00 = wy_inv * wx_inv
                w01 = wy_inv * wx
                w10 = wy * wx_inv
                w11 = wy * wx

                do c = 1, 3
                    dst_f32(c, dx, dy) = &
                        src_f32(c, x0, y0) * w00 + src_f32(c, x1, y0) * w01 + &
                        src_f32(c, x0, y1) * w10 + src_f32(c, x1, y1) * w11
                end do
            end do
        end do
    end subroutine kernel_bilinear_resize_rgb

    subroutine kernel_letterbox_pad(src_hwc, rw, rh, pad_x, pad_y, out_hwc)
        real(c_float), intent(in) :: src_hwc(:, :, :)
        integer(int32), intent(in) :: rw, rh, pad_x, pad_y
        real(c_float), intent(out) :: out_hwc(:, :, :)
        integer(int32) :: x, y

        out_hwc(:, :, :) = PAD_VALUE_F32
        do y = 1, rh
            do x = 1, rw
                out_hwc(:, pad_x + x, pad_y + y) = src_hwc(:, x, y)
            end do
        end do
    end subroutine kernel_letterbox_pad

    subroutine kernel_hwc_to_chw(hwc, w, h, chw)
        real(c_float), intent(in) :: hwc(:, :, :)
        integer(int32), intent(in) :: w, h
        real(c_float), intent(out) :: chw(:, :, :)
        integer(int32) :: x, y, c

        do c = 1, 3
            do y = 1, h
                do x = 1, w
                    chw(x, y, c) = hwc(c, x, y)
                end do
            end do
        end do
    end subroutine kernel_hwc_to_chw

end module preprocessing_mod
