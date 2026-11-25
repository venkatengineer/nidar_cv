#ifndef CAMERA_V4L2_H
#define CAMERA_V4L2_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CAM_PIXFMT_YUYV = 0,
    CAM_PIXFMT_MJPG = 1,
} CamPixFmt;

typedef struct {
    int fd;
    uint32_t width;
    uint32_t height;
    CamPixFmt pixfmt;
    uint32_t buffer_count;
    void* buffers[8];
    size_t buffer_lengths[8];
    int streaming;
} V4L2Camera;

typedef struct {
    const uint8_t* data;
    size_t size;
    uint32_t index;
} V4L2Frame;

int v4l2_cam_open(V4L2Camera* cam, const char* device_path, uint32_t width, uint32_t height, CamPixFmt pixfmt);
int v4l2_cam_dequeue(V4L2Camera* cam, V4L2Frame* out_frame, int timeout_ms);
int v4l2_cam_enqueue(V4L2Camera* cam, uint32_t buffer_index);
void v4l2_cam_close(V4L2Camera* cam);

#ifdef __cplusplus
}
#endif

#endif
