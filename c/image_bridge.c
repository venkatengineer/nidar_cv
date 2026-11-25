#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_FAILURE_STRINGS
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdint.h>

uint8_t* img_bridge_load_rgb(const char* path, int32_t* out_w, int32_t* out_h) {
    int w = 0;
    int h = 0;
    int comp = 0;
    unsigned char* ptr = stbi_load(path, &w, &h, &comp, 3);
    if (ptr == NULL || w <= 0 || h <= 0) {
        if (ptr != NULL) stbi_image_free(ptr);
        *out_w = 0;
        *out_h = 0;
        return NULL;
    }
    *out_w = (int32_t)w;
    *out_h = (int32_t)h;
    return (uint8_t*)ptr;
}

void img_bridge_free(uint8_t* ptr) {
    if (ptr != NULL) {
        stbi_image_free(ptr);
    }
}

int32_t img_bridge_save_png(const char* path, int32_t w, int32_t h, const uint8_t* rgb_data) {
    const int stride = (int)w * 3;
    return (int32_t)stbi_write_png(path, (int)w, (int)h, 3, rgb_data, stride);
}
