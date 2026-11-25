#include "image.hpp"

#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace yolo {

namespace {

uint32_t clampCoord(float val, uint32_t max_val) {
    if (val <= 0.0f) return 0;
    const float fmax = static_cast<float>(max_val);
    if (val >= fmax) return max_val;
    return static_cast<uint32_t>(val);
}

std::array<uint8_t, 3> classColor(uint32_t class_id) {
    constexpr std::array<std::array<uint8_t, 3>, 8> palette = {{
        {220, 50, 47},
        {38, 139, 210},
        {133, 153, 0},
        {211, 54, 130},
        {42, 161, 152},
        {203, 75, 22},
        {108, 113, 196},
        {181, 137, 0},
    }};
    return palette[class_id % palette.size()];
}

std::array<uint8_t, 5> asciiGlyph5x7(char ch_in) {
    const unsigned char ch = static_cast<unsigned char>(std::toupper(static_cast<unsigned char>(ch_in)));
    switch (ch) {
        case 'A': return {0x7E, 0x11, 0x11, 0x11, 0x7E};
        case 'B': return {0x7F, 0x49, 0x49, 0x49, 0x36};
        case 'C': return {0x3E, 0x41, 0x41, 0x41, 0x22};
        case 'D': return {0x7F, 0x41, 0x41, 0x22, 0x1C};
        case 'E': return {0x7F, 0x49, 0x49, 0x49, 0x41};
        case 'F': return {0x7F, 0x09, 0x09, 0x09, 0x01};
        case 'G': return {0x3E, 0x41, 0x49, 0x49, 0x7A};
        case 'H': return {0x7F, 0x08, 0x08, 0x08, 0x7F};
        case 'I': return {0x00, 0x41, 0x7F, 0x41, 0x00};
        case 'J': return {0x20, 0x40, 0x41, 0x3F, 0x01};
        case 'K': return {0x7F, 0x08, 0x14, 0x22, 0x41};
        case 'L': return {0x7F, 0x40, 0x40, 0x40, 0x40};
        case 'M': return {0x7F, 0x02, 0x0C, 0x02, 0x7F};
        case 'N': return {0x7F, 0x04, 0x08, 0x10, 0x7F};
        case 'O': return {0x3E, 0x41, 0x41, 0x41, 0x3E};
        case 'P': return {0x7F, 0x09, 0x09, 0x09, 0x06};
        case 'Q': return {0x3E, 0x41, 0x51, 0x21, 0x5E};
        case 'R': return {0x7F, 0x09, 0x19, 0x29, 0x46};
        case 'S': return {0x46, 0x49, 0x49, 0x49, 0x31};
        case 'T': return {0x01, 0x01, 0x7F, 0x01, 0x01};
        case 'U': return {0x3F, 0x40, 0x40, 0x40, 0x3F};
        case 'V': return {0x1F, 0x20, 0x40, 0x20, 0x1F};
        case 'W': return {0x3F, 0x40, 0x38, 0x40, 0x3F};
        case 'X': return {0x63, 0x14, 0x08, 0x14, 0x63};
        case 'Y': return {0x07, 0x08, 0x70, 0x08, 0x07};
        case 'Z': return {0x61, 0x51, 0x49, 0x45, 0x43};
        case '0': return {0x3E, 0x51, 0x49, 0x45, 0x3E};
        case '1': return {0x00, 0x42, 0x7F, 0x40, 0x00};
        case '2': return {0x42, 0x61, 0x51, 0x49, 0x46};
        case '3': return {0x21, 0x41, 0x45, 0x4B, 0x31};
        case '4': return {0x18, 0x14, 0x12, 0x7F, 0x10};
        case '5': return {0x27, 0x45, 0x45, 0x45, 0x39};
        case '6': return {0x3C, 0x4A, 0x49, 0x49, 0x30};
        case '7': return {0x01, 0x71, 0x09, 0x05, 0x03};
        case '8': return {0x36, 0x49, 0x49, 0x49, 0x36};
        case '9': return {0x06, 0x49, 0x49, 0x29, 0x1E};
        case '%': return {0x23, 0x13, 0x08, 0x64, 0x62};
        case '-': return {0x08, 0x08, 0x08, 0x08, 0x08};
        case '_': return {0x40, 0x40, 0x40, 0x40, 0x40};
        case '.': return {0x00, 0x60, 0x60, 0x00, 0x00};
        case ' ': return {0x00, 0x00, 0x00, 0x00, 0x00};
        default: return {0x00, 0x00, 0x00, 0x00, 0x00};
    }
}

} // namespace

Image::~Image() {
    if (owns_stb_ && data_ != nullptr) {
        stbi_image_free(data_);
    }
    data_ = nullptr;
}

Image::Image(Image&& other) noexcept
    : width_(other.width_),
      height_(other.height_),
      channels_(other.channels_),
      data_(other.data_),
      owns_stb_(other.owns_stb_),
      owned_buf_(std::move(other.owned_buf_)) {
    if (!owns_stb_ && !owned_buf_.empty()) {
        data_ = owned_buf_.data();
    }
    other.width_ = 0;
    other.height_ = 0;
    other.channels_ = 0;
    other.data_ = nullptr;
    other.owns_stb_ = false;
}

Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        if (owns_stb_ && data_ != nullptr) {
            stbi_image_free(data_);
        }
        width_ = other.width_;
        height_ = other.height_;
        channels_ = other.channels_;
        owns_stb_ = other.owns_stb_;
        owned_buf_ = std::move(other.owned_buf_);
        data_ = owns_stb_ ? other.data_ : (owned_buf_.empty() ? nullptr : owned_buf_.data());

        other.width_ = 0;
        other.height_ = 0;
        other.channels_ = 0;
        other.data_ = nullptr;
        other.owns_stb_ = false;
    }
    return *this;
}

Image Image::load(const std::string& path) {
    int w = 0;
    int h = 0;
    int orig_channels = 0;
    uint8_t* ptr = stbi_load(path.c_str(), &w, &h, &orig_channels, 3);
    if (ptr == nullptr || w <= 0 || h <= 0) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    Image img;
    img.width_ = static_cast<uint32_t>(w);
    img.height_ = static_cast<uint32_t>(h);
    img.channels_ = 3;
    img.data_ = ptr;
    img.owns_stb_ = true;
    return img;
}

Image Image::clone() const {
    Image copy;
    copy.width_ = width_;
    copy.height_ = height_;
    copy.channels_ = channels_;
    copy.owns_stb_ = false;
    copy.owned_buf_.assign(data_, data_ + byteLen());
    copy.data_ = copy.owned_buf_.data();
    return copy;
}

void Image::savePng(const std::string& path) const {
    const int stride = static_cast<int>(width_ * channels_);
    const int ok = stbi_write_png(
        path.c_str(),
        static_cast<int>(width_),
        static_cast<int>(height_),
        static_cast<int>(channels_),
        data_,
        stride);
    if (ok == 0) {
        throw std::runtime_error("Failed to write PNG: " + path);
    }
}

void Image::drawDetections(const Detection* detections, size_t count) {
    if (width_ == 0 || height_ == 0) return;
    for (size_t i = 0; i < count; ++i) {
        const Detection& det = detections[i];
        const auto color = classColor(det.class_id);
        const uint32_t x1 = clampCoord(det.x1, width_ - 1);
        const uint32_t y1 = clampCoord(det.y1, height_ - 1);
        const uint32_t x2 = clampCoord(det.x2, width_ - 1);
        const uint32_t y2 = clampCoord(det.y2, height_ - 1);
        if (x2 <= x1 || y2 <= y1) continue;

        drawRectOutline(x1, y1, x2, y2, 3, color.data());

        const std::string_view cname = className(det.class_id);
        const uint32_t conf_pct = static_cast<uint32_t>(std::round(std::clamp(det.confidence, 0.0f, 1.0f) * 100.0f));
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.*s %u%%", static_cast<int>(cname.size()), cname.data(), conf_pct);
        const std::string label(buf);

        constexpr uint32_t scale = 2;
        const uint32_t text_w = static_cast<uint32_t>(label.size()) * 6 * scale + 6;
        const uint32_t text_h = 7 * scale + 6;
        const uint32_t label_y = (y1 >= text_h) ? (y1 - text_h) : y1;
        const uint32_t label_x2 = std::min(x1 + text_w, width_ - 1);
        const uint32_t label_y2 = std::min(label_y + text_h, height_ - 1);

        fillRect(x1, label_y, label_x2, label_y2, color.data());
        constexpr uint8_t white[3] = {255, 255, 255};
        drawText(x1 + 3, label_y + 3, label, scale, white);
    }
}

void Image::setPixel(uint32_t x, uint32_t y, const uint8_t color[3]) {
    if (x >= width_ || y >= height_) return;
    const size_t idx = (static_cast<size_t>(y) * width_ + x) * 3;
    data_[idx + 0] = color[0];
    data_[idx + 1] = color[1];
    data_[idx + 2] = color[2];
}

void Image::fillRect(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, const uint8_t color[3]) {
    for (uint32_t y = y1; y <= y2; ++y) {
        for (uint32_t x = x1; x <= x2; ++x) {
            setPixel(x, y, color);
        }
    }
}

void Image::drawRectOutline(
    uint32_t x1,
    uint32_t y1,
    uint32_t x2,
    uint32_t y2,
    uint32_t thickness,
    const uint8_t color[3]) {
    for (uint32_t t = 0; t < thickness; ++t) {
        const uint32_t tx1 = (x1 + t <= x2) ? (x1 + t) : x2;
        const uint32_t ty1 = (y1 + t <= y2) ? (y1 + t) : y2;
        const uint32_t tx2 = (x2 >= x1 + t) ? (x2 - t) : x1;
        const uint32_t ty2 = (y2 >= y1 + t) ? (y2 - t) : y1;

        for (uint32_t x = tx1; x <= tx2; ++x) {
            setPixel(x, ty1, color);
            setPixel(x, ty2, color);
        }
        for (uint32_t y = ty1; y <= ty2; ++y) {
            setPixel(tx1, y, color);
            setPixel(tx2, y, color);
        }
    }
}

void Image::drawText(
    uint32_t x,
    uint32_t y,
    const std::string& text,
    uint32_t scale,
    const uint8_t color[3]) {
    uint32_t cx = x;
    for (char ch : text) {
        const auto glyph = asciiGlyph5x7(ch);
        for (uint32_t col = 0; col < 5; ++col) {
            const uint8_t bits = glyph[col];
            for (uint32_t row = 0; row < 7; ++row) {
                if ((bits & (1u << row)) != 0) {
                    for (uint32_t sy = 0; sy < scale; ++sy) {
                        for (uint32_t sx = 0; sx < scale; ++sx) {
                            setPixel(cx + col * scale + sx, y + row * scale + sy, color);
                        }
                    }
                }
            }
        }
        cx += 6 * scale;
    }
}

} // namespace yolo
