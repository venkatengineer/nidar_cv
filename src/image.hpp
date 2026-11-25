#pragma once

#include "postprocessing.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace yolo {

class Image {
public:
    Image() = default;
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    static Image load(const std::string& path);
    Image clone() const;

    void savePng(const std::string& path) const;
    void drawDetections(const Detection* detections, size_t count);

    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    uint32_t channels() const { return channels_; }
    const uint8_t* data() const { return data_; }
    size_t byteLen() const {
        return static_cast<size_t>(width_) * static_cast<size_t>(height_) * channels_;
    }

private:
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t channels_ = 0;
    uint8_t* data_ = nullptr;
    bool owns_stb_ = false;
    std::vector<uint8_t> owned_buf_;

    void setPixel(uint32_t x, uint32_t y, const uint8_t color[3]);
    void fillRect(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, const uint8_t color[3]);
    void drawRectOutline(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t thickness, const uint8_t color[3]);
    void drawText(uint32_t x, uint32_t y, const std::string& text, uint32_t scale, const uint8_t color[3]);
};

} // namespace yolo
