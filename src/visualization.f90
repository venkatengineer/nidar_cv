module visualization_mod
    use, intrinsic :: iso_c_binding, only: c_int8_t
    use, intrinsic :: iso_fortran_env, only: int32, real32
    use image_mod, only: image_t
    use postprocessing_mod, only: detection_t, coco_class_name
    implicit none
    private

    public :: draw_detections

    integer(int32), parameter :: PALETTE(3, 8) = reshape([ &
        220, 50, 47,   &
        38, 139, 210,  &
        133, 153, 0,   &
        211, 54, 130,  &
        42, 161, 152,  &
        203, 75, 22,   &
        108, 113, 196, &
        181, 137, 0    &
    ], [3, 8])

contains

    pure function u8_byte(val) result(b)
        integer(int32), intent(in) :: val
        integer(c_int8_t) :: b
        integer(int32) :: v
        v = min(max(val, 0), 255)
        if (v >= 128) then
            b = int(v - 256, c_int8_t)
        else
            b = int(v, c_int8_t)
        end if
    end function u8_byte

    pure function clamp_coord(val, max_val) result(res)
        real(real32), intent(in) :: val
        integer(int32), intent(in) :: max_val
        integer(int32) :: res
        if (val <= 0.0_real32) then
            res = 1
        else if (val >= real(max_val - 1, real32)) then
            res = max_val
        else
            res = int(val, int32) + 1
        end if
    end function clamp_coord

    subroutine set_pixel(img, x, y, color)
        type(image_t), intent(inout) :: img
        integer(int32), intent(in) :: x, y, color(3)
        if (x < 1 .or. x > img%width .or. y < 1 .or. y > img%height) return
        img%view(1, x, y) = u8_byte(color(1))
        img%view(2, x, y) = u8_byte(color(2))
        img%view(3, x, y) = u8_byte(color(3))
    end subroutine set_pixel

    subroutine fill_rect(img, x1, y1, x2, y2, color)
        type(image_t), intent(inout) :: img
        integer(int32), intent(in) :: x1, y1, x2, y2, color(3)
        integer(int32) :: x, y
        do y = y1, y2
            do x = x1, x2
                call set_pixel(img, x, y, color)
            end do
        end do
    end subroutine fill_rect

    subroutine draw_rect_outline(img, x1, y1, x2, y2, thickness, color)
        type(image_t), intent(inout) :: img
        integer(int32), intent(in) :: x1, y1, x2, y2, thickness, color(3)
        integer(int32) :: t, tx1, ty1, tx2, ty2, x, y

        do t = 0, thickness - 1
            tx1 = min(x1 + t, x2)
            ty1 = min(y1 + t, y2)
            tx2 = max(x2 - t, x1)
            ty2 = max(y2 - t, y1)
            do x = tx1, tx2
                call set_pixel(img, x, ty1, color)
                call set_pixel(img, x, ty2, color)
            end do
            do y = ty1, ty2
                call set_pixel(img, tx1, y, color)
                call set_pixel(img, tx2, y, color)
            end do
        end do
    end subroutine draw_rect_outline

    pure function ascii_glyph_5x7(ch_in) result(g)
        character(len=1), intent(in) :: ch_in
        integer(int32) :: g(5), code
        character(len=1) :: ch

        code = iachar(ch_in)
        if (code >= iachar('a') .and. code <= iachar('z')) then
            ch = achar(code - 32)
        else
            ch = ch_in
        end if

        select case (ch)
        case ('A'); g = [int(z'7E'), int(z'11'), int(z'11'), int(z'11'), int(z'7E')]
        case ('B'); g = [int(z'7F'), int(z'49'), int(z'49'), int(z'49'), int(z'36')]
        case ('C'); g = [int(z'3E'), int(z'41'), int(z'41'), int(z'41'), int(z'22')]
        case ('D'); g = [int(z'7F'), int(z'41'), int(z'41'), int(z'22'), int(z'1C')]
        case ('E'); g = [int(z'7F'), int(z'49'), int(z'49'), int(z'49'), int(z'41')]
        case ('F'); g = [int(z'7F'), int(z'09'), int(z'09'), int(z'09'), int(z'01')]
        case ('G'); g = [int(z'3E'), int(z'41'), int(z'49'), int(z'49'), int(z'7A')]
        case ('H'); g = [int(z'7F'), int(z'08'), int(z'08'), int(z'08'), int(z'7F')]
        case ('I'); g = [int(z'00'), int(z'41'), int(z'7F'), int(z'41'), int(z'00')]
        case ('J'); g = [int(z'20'), int(z'40'), int(z'41'), int(z'3F'), int(z'01')]
        case ('K'); g = [int(z'7F'), int(z'08'), int(z'14'), int(z'22'), int(z'41')]
        case ('L'); g = [int(z'7F'), int(z'40'), int(z'40'), int(z'40'), int(z'40')]
        case ('M'); g = [int(z'7F'), int(z'02'), int(z'0C'), int(z'02'), int(z'7F')]
        case ('N'); g = [int(z'7F'), int(z'04'), int(z'08'), int(z'10'), int(z'7F')]
        case ('O'); g = [int(z'3E'), int(z'41'), int(z'41'), int(z'41'), int(z'3E')]
        case ('P'); g = [int(z'7F'), int(z'09'), int(z'09'), int(z'09'), int(z'06')]
        case ('Q'); g = [int(z'3E'), int(z'41'), int(z'51'), int(z'21'), int(z'5E')]
        case ('R'); g = [int(z'7F'), int(z'09'), int(z'19'), int(z'29'), int(z'46')]
        case ('S'); g = [int(z'46'), int(z'49'), int(z'49'), int(z'49'), int(z'31')]
        case ('T'); g = [int(z'01'), int(z'01'), int(z'7F'), int(z'01'), int(z'01')]
        case ('U'); g = [int(z'3F'), int(z'40'), int(z'40'), int(z'40'), int(z'3F')]
        case ('V'); g = [int(z'1F'), int(z'20'), int(z'40'), int(z'20'), int(z'1F')]
        case ('W'); g = [int(z'3F'), int(z'40'), int(z'38'), int(z'40'), int(z'3F')]
        case ('X'); g = [int(z'63'), int(z'14'), int(z'08'), int(z'14'), int(z'63')]
        case ('Y'); g = [int(z'07'), int(z'08'), int(z'70'), int(z'08'), int(z'07')]
        case ('Z'); g = [int(z'61'), int(z'51'), int(z'49'), int(z'45'), int(z'43')]
        case ('0'); g = [int(z'3E'), int(z'51'), int(z'49'), int(z'45'), int(z'3E')]
        case ('1'); g = [int(z'00'), int(z'42'), int(z'7F'), int(z'40'), int(z'00')]
        case ('2'); g = [int(z'42'), int(z'61'), int(z'51'), int(z'49'), int(z'46')]
        case ('3'); g = [int(z'21'), int(z'41'), int(z'45'), int(z'4B'), int(z'31')]
        case ('4'); g = [int(z'18'), int(z'14'), int(z'12'), int(z'7F'), int(z'10')]
        case ('5'); g = [int(z'27'), int(z'45'), int(z'45'), int(z'45'), int(z'39')]
        case ('6'); g = [int(z'3C'), int(z'4A'), int(z'49'), int(z'49'), int(z'30')]
        case ('7'); g = [int(z'01'), int(z'71'), int(z'09'), int(z'05'), int(z'03')]
        case ('8'); g = [int(z'36'), int(z'49'), int(z'49'), int(z'49'), int(z'36')]
        case ('9'); g = [int(z'06'), int(z'49'), int(z'49'), int(z'29'), int(z'1E')]
        case ('%'); g = [int(z'23'), int(z'13'), int(z'08'), int(z'64'), int(z'62')]
        case ('-'); g = [int(z'08'), int(z'08'), int(z'08'), int(z'08'), int(z'08')]
        case ('_'); g = [int(z'40'), int(z'40'), int(z'40'), int(z'40'), int(z'40')]
        case ('.'); g = [int(z'00'), int(z'60'), int(z'60'), int(z'00'), int(z'00')]
        case default; g = 0
        end select
    end function ascii_glyph_5x7

    subroutine draw_text(img, x, y, text, scale, color)
        type(image_t), intent(inout) :: img
        integer(int32), intent(in) :: x, y, scale, color(3)
        character(len=*), intent(in) :: text
        integer(int32) :: i, col, row, sx, sy, cx, g(5)

        cx = x
        do i = 1, len_trim(text)
            g = ascii_glyph_5x7(text(i:i))
            do col = 0, 4
                do row = 0, 6
                    if (btest(g(col + 1), row)) then
                        do sy = 0, scale - 1
                            do sx = 0, scale - 1
                                call set_pixel(img, cx + col * scale + sx, y + row * scale + sy, color)
                            end do
                        end do
                    end if
                end do
            end do
            cx = cx + 6 * scale
        end do
    end subroutine draw_text

    subroutine draw_detections(img, detections, count)
        type(image_t), intent(inout) :: img
        type(detection_t), intent(in) :: detections(:)
        integer(int32), intent(in) :: count

        integer(int32) :: i, x1, y1, x2, y2, color(3), conf_pct
        integer(int32) :: scale, text_w, text_h, label_y, label_x2, label_y2
        integer(int32), parameter :: WHITE(3) = [255, 255, 255]
        character(len=64) :: label
        character(len=:), allocatable :: cname

        if (img%width <= 0 .or. img%height <= 0) return

        do i = 1, count
            color = PALETTE(:, mod(detections(i)%class_id, 8) + 1)
            x1 = clamp_coord(detections(i)%x1, img%width)
            y1 = clamp_coord(detections(i)%y1, img%height)
            x2 = clamp_coord(detections(i)%x2, img%width)
            y2 = clamp_coord(detections(i)%y2, img%height)
            if (x2 <= x1 .or. y2 <= y1) cycle

            call draw_rect_outline(img, x1, y1, x2, y2, 3, color)

            cname = coco_class_name(detections(i)%class_id)
            conf_pct = nint(min(max(detections(i)%confidence, 0.0_real32), 1.0_real32) * 100.0_real32)
            write(label, '(A,1X,I0,A)') cname, conf_pct, "%"

            scale = 2
            text_w = len_trim(label) * 6 * scale + 6
            text_h = 7 * scale + 6
            if (y1 > text_h) then
                label_y = y1 - text_h
            else
                label_y = y1
            end if
            label_x2 = min(x1 + text_w, img%width)
            label_y2 = min(label_y + text_h, img%height)

            call fill_rect(img, x1, label_y, label_x2, label_y2, color)
            call draw_text(img, x1 + 3, label_y + 3, label, scale, WHITE)
        end do
    end subroutine draw_detections

end module visualization_mod
