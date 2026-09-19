#include "pisa_vae_segment_pack.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <utility>

namespace kira::pisa {
namespace {

constexpr std::uint8_t MAGIC[] = {
    'K', 'R', 'V', 'S', 'E', 'G', '0', '1',
};
constexpr std::uint32_t VERSION = 1;
constexpr std::size_t MAX_SEGMENTS_PER_SIDE = 128;
constexpr std::size_t MAX_CHANNELS = 8192;

class Reader {
public:
    Reader(const std::uint8_t* data, std::size_t size)
        : data_(data), size_(size) {}

    bool bytes(std::size_t count, const std::uint8_t*& output) {
        if (
            data_ == nullptr ||
            count > size_ - offset_
        ) {
            return false;
        }
        output = data_ + offset_;
        offset_ += count;
        return true;
    }

    bool u32(std::uint32_t& output) {
        const std::uint8_t* ptr = nullptr;
        if (!bytes(4, ptr)) {
            return false;
        }
        output =
            static_cast<std::uint32_t>(ptr[0]) |
            (static_cast<std::uint32_t>(ptr[1]) << 8U) |
            (static_cast<std::uint32_t>(ptr[2]) << 16U) |
            (static_cast<std::uint32_t>(ptr[3]) << 24U);
        return true;
    }

    bool u64(std::uint64_t& output) {
        const std::uint8_t* ptr = nullptr;
        if (!bytes(8, ptr)) {
            return false;
        }
        output = 0;
        for (int index = 0; index < 8; ++index) {
            output |=
                static_cast<std::uint64_t>(ptr[index]) <<
                (static_cast<unsigned>(index) * 8U);
        }
        return true;
    }

    bool f32(float& output) {
        std::uint32_t bits = 0;
        if (!u32(bits)) {
            return false;
        }
        static_assert(sizeof(bits) == sizeof(output));
        std::memcpy(&output, &bits, sizeof(output));
        return std::isfinite(output);
    }

    bool atEnd() const {
        return offset_ == size_;
    }

private:
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t offset_ = 0;
};

bool readAffine(
    Reader& reader,
    VaeGroupNormAffine& output
) {
    std::uint32_t groups = 0;
    std::uint32_t channels = 0;
    float epsilon = 0.0f;
    if (
        !reader.u32(groups) ||
        !reader.u32(channels) ||
        !reader.f32(epsilon) ||
        groups == 0 ||
        channels == 0 ||
        channels > MAX_CHANNELS ||
        channels % groups != 0 ||
        epsilon <= 0.0f
    ) {
        return false;
    }

    VaeGroupNormAffine result;
    result.groups = static_cast<int>(groups);
    result.channels = static_cast<int>(channels);
    result.epsilon = epsilon;
    result.weight.resize(channels);
    result.bias.resize(channels);

    for (float& value : result.weight) {
        if (!reader.f32(value)) {
            return false;
        }
    }
    for (float& value : result.bias) {
        if (!reader.f32(value)) {
            return false;
        }
    }

    output = std::move(result);
    return true;
}

bool readModel(
    Reader& reader,
    std::vector<std::uint8_t>& output
) {
    std::uint64_t size64 = 0;
    if (
        !reader.u64(size64) ||
        size64 == 0 ||
        size64 >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()
            )
    ) {
        return false;
    }

    const auto size = static_cast<std::size_t>(size64);
    const std::uint8_t* bytes = nullptr;
    if (!reader.bytes(size, bytes)) {
        return false;
    }
    output.assign(bytes, bytes + size);
    return true;
}

bool validCounts(
    std::uint32_t segmentCount,
    std::uint32_t normCount
) {
    return
        segmentCount > 0 &&
        segmentCount <= MAX_SEGMENTS_PER_SIDE &&
        normCount + 1 == segmentCount;
}

}  // namespace

bool parseVaeSegmentPack(
    const std::uint8_t* data,
    std::size_t size,
    VaeSegmentPack& output
) {
    output = VaeSegmentPack{};
    if (
        data == nullptr ||
        size < sizeof(MAGIC) + 5U * sizeof(std::uint32_t)
    ) {
        return false;
    }

    Reader reader(data, size);
    const std::uint8_t* magic = nullptr;
    if (
        !reader.bytes(sizeof(MAGIC), magic) ||
        std::memcmp(magic, MAGIC, sizeof(MAGIC)) != 0
    ) {
        return false;
    }

    std::uint32_t version = 0;
    std::uint32_t encoderSegments = 0;
    std::uint32_t decoderSegments = 0;
    std::uint32_t encoderNorms = 0;
    std::uint32_t decoderNorms = 0;
    if (
        !reader.u32(version) ||
        !reader.u32(encoderSegments) ||
        !reader.u32(decoderSegments) ||
        !reader.u32(encoderNorms) ||
        !reader.u32(decoderNorms) ||
        version != VERSION ||
        !validCounts(encoderSegments, encoderNorms) ||
        !validCounts(decoderSegments, decoderNorms)
    ) {
        return false;
    }

    VaeSegmentPack result;
    result.encoderAffine.resize(encoderNorms);
    result.decoderAffine.resize(decoderNorms);
    result.encoderModels.resize(encoderSegments);
    result.decoderModels.resize(decoderSegments);

    for (auto& affine : result.encoderAffine) {
        if (!readAffine(reader, affine)) {
            return false;
        }
    }
    for (auto& affine : result.decoderAffine) {
        if (!readAffine(reader, affine)) {
            return false;
        }
    }
    for (auto& model : result.encoderModels) {
        if (!readModel(reader, model)) {
            return false;
        }
    }
    for (auto& model : result.decoderModels) {
        if (!readModel(reader, model)) {
            return false;
        }
    }

    if (!reader.atEnd()) {
        return false;
    }

    output = std::move(result);
    return true;
}

bool loadVaeSegmentPackFile(
    const std::string& path,
    VaeSegmentPack& output
) {
    output = VaeSegmentPack{};
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return false;
    }

    const std::streamoff end = input.tellg();
    if (
        end <= 0 ||
        end >
            static_cast<std::streamoff>(
                std::numeric_limits<std::streamsize>::max()
            )
    ) {
        return false;
    }
    const auto size = static_cast<std::size_t>(end);
    std::vector<std::uint8_t> bytes(size);
    input.seekg(0, std::ios::beg);
    if (
        !input ||
        !input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(size)
        )
    ) {
        return false;
    }

    return parseVaeSegmentPack(
        bytes.data(),
        bytes.size(),
        output
    );
}

}  // namespace kira::pisa
