#include "camera_v4l2.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

static int xioctl(int fd, unsigned long request, void* arg) {
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

int v4l2_cam_open(V4L2Camera* cam, const char* device_path, uint32_t width, uint32_t height, CamPixFmt pixfmt) {
    memset(cam, 0, sizeof(*cam));
    cam->fd = -1;

    int fd = open(device_path, O_RDWR | O_NONBLOCK, 0);
    if (fd < 0) {
        return -1;
    }

    struct v4l2_capability cap;
    memset(&cap, 0, sizeof(cap));
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
        close(fd);
        return -2;
    }
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) || !(cap.capabilities & V4L2_CAP_STREAMING)) {
        close(fd);
        return -3;
    }

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = (pixfmt == CAM_PIXFMT_MJPG) ? V4L2_PIX_FMT_MJPEG : V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;

    if (xioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        close(fd);
        return -4;
    }

    cam->fd = fd;
    cam->width = fmt.fmt.pix.width;
    cam->height = fmt.fmt.pix.height;
    cam->pixfmt = (fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_MJPEG) ? CAM_PIXFMT_MJPG : CAM_PIXFMT_YUYV;

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (xioctl(fd, VIDIOC_REQBUFS, &req) < 0 || req.count < 2) {
        v4l2_cam_close(cam);
        return -5;
    }
    if (req.count > 8) req.count = 8;

    for (uint32_t i = 0; i < req.count; ++i) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (xioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
            v4l2_cam_close(cam);
            return -6;
        }

        void* ptr = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if (ptr == MAP_FAILED) {
            v4l2_cam_close(cam);
            return -7;
        }
        cam->buffers[i] = ptr;
        cam->buffer_lengths[i] = buf.length;
        cam->buffer_count = i + 1;

        if (xioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            v4l2_cam_close(cam);
            return -8;
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_STREAMON, &type) < 0) {
        v4l2_cam_close(cam);
        return -9;
    }
    cam->streaming = 1;
    return 0;
}

int v4l2_cam_dequeue(V4L2Camera* cam, V4L2Frame* out_frame, int timeout_ms) {
    struct pollfd pfd;
    pfd.fd = cam->fd;
    pfd.events = POLLIN;
    pfd.revents = 0;

    int pr;
    do {
        pr = poll(&pfd, 1, timeout_ms);
    } while (pr < 0 && errno == EINTR);

    if (pr == 0) return 1; // timeout
    if (pr < 0) return -1;

    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (xioctl(cam->fd, VIDIOC_DQBUF, &buf) < 0) {
        if (errno == EAGAIN) return 1;
        return -2;
    }

    out_frame->data = (const uint8_t*)cam->buffers[buf.index];
    out_frame->size = buf.bytesused;
    out_frame->index = buf.index;
    return 0;
}

int v4l2_cam_enqueue(V4L2Camera* cam, uint32_t buffer_index) {
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = buffer_index;
    return xioctl(cam->fd, VIDIOC_QBUF, &buf);
}

void v4l2_cam_close(V4L2Camera* cam) {
    if (cam->fd >= 0) {
        if (cam->streaming) {
            enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            xioctl(cam->fd, VIDIOC_STREAMOFF, &type);
            cam->streaming = 0;
        }
        for (uint32_t i = 0; i < cam->buffer_count; ++i) {
            if (cam->buffers[i] != NULL) {
                munmap(cam->buffers[i], cam->buffer_lengths[i]);
                cam->buffers[i] = NULL;
            }
        }
        close(cam->fd);
        cam->fd = -1;
    }
}
