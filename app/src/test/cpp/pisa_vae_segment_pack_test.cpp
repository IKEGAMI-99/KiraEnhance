#include "pisa_vae_segment_pack.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void expectTrue(bool value, const char* label) {
    if (!value) {
        std::cerr << "FAIL: " << label << "\n";
        ++failures;
    }
}

void expectFalse(bool value, const char* label) {
    expectTrue(!value, label);
}

void appendU32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        bytes.push_back(
            static_cast<std::uint8_t>((value >> shift) & 0xffU)
        );
    }
}

void appendU64(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        bytes.push_back(
            static_cast<std::uint8_t>((value >> shift) & 0xffULL)
        );
    }
}

void appendF32(std::vector<std::uint8_t>& bytes, float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(value));
    appendU32(bytes, bits);
}

void appendAffine(
    std::vector<std::uint8_t>& bytes,
    float epsilon,
    float firstWeight
) {
    appendU32(bytes, 1);
    appendU32(bytes, 2);
    appendF32(bytes, epsilon);
    appendF32(bytes, firstWeight);
    appendF32(bytes, firstWeight + 1.0f);
    appendF32(bytes, -1.0f);
    appendF32(bytes, 1.0f);
}

void appendModel(
    std::vector<std::uint8_t>& bytes,
    const std::vector<std::uint8_t>& model
) {
    appendU64(bytes, model.size());
    bytes.insert(bytes.end(), model.begin(), model.end());
}

std::vector<std::uint8_t> validPack() {
    std::vector<std::uint8_t> bytes = {
        'K', 'R', 'V', 'S', 'E', 'G', '0', '1',
    };
    appendU32(bytes, 1);
    appendU32(bytes, 2);
    appendU32(bytes, 2);
    appendU32(bytes, 1);
    appendU32(bytes, 1);

    appendAffine(bytes, 1.0e-6f, 2.0f);
    appendAffine(bytes, 2.0e-6f, 4.0f);

    appendModel(bytes, {1, 2, 3});
    appendModel(bytes, {4, 5});
    appendModel(bytes, {6});
    appendModel(bytes, {7, 8, 9, 10});
    return bytes;
}

void testParsesValidPack() {
    const auto bytes = validPack();
    kira::pisa::VaeSegmentPack pack;
    expectTrue(
        kira::pisa::parseVaeSegmentPack(
            bytes.data(),
            bytes.size(),
            pack
        ),
        "valid pack parses"
    );
    expectTrue(pack.encoderModels.size() == 2, "encoder model count");
    expectTrue(pack.decoderModels.size() == 2, "decoder model count");
    expectTrue(pack.encoderAffine.size() == 1, "encoder affine count");
    expectTrue(pack.decoderAffine.size() == 1, "decoder affine count");
    expectTrue(pack.encoderModels[0].size() == 3, "first model bytes");
    expectTrue(pack.decoderModels[1].size() == 4, "last model bytes");
    expectTrue(pack.encoderAffine[0].channels == 2, "affine channels");
    expectTrue(
        pack.encoderAffine[0].weight[0] == 2.0f,
        "affine weight"
    );
}

void testRejectsBadMagicAndTrailingBytes() {
    auto badMagic = validPack();
    badMagic[0] = 'X';
    kira::pisa::VaeSegmentPack pack;
    expectFalse(
        kira::pisa::parseVaeSegmentPack(
            badMagic.data(),
            badMagic.size(),
            pack
        ),
        "bad magic rejected"
    );

    auto trailing = validPack();
    trailing.push_back(0);
    expectFalse(
        kira::pisa::parseVaeSegmentPack(
            trailing.data(),
            trailing.size(),
            pack
        ),
        "trailing bytes rejected"
    );
}

void testRejectsTruncatedModelAndInvalidCounts() {
    auto truncated = validPack();
    truncated.pop_back();
    kira::pisa::VaeSegmentPack pack;
    expectFalse(
        kira::pisa::parseVaeSegmentPack(
            truncated.data(),
            truncated.size(),
            pack
        ),
        "truncated model rejected"
    );

    auto counts = validPack();
    counts[12] = 3;
    expectFalse(
        kira::pisa::parseVaeSegmentPack(
            counts.data(),
            counts.size(),
            pack
        ),
        "segment norm count mismatch rejected"
    );
}

}  // namespace

int main() {
    testParsesValidPack();
    testRejectsBadMagicAndTrailingBytes();
    testRejectsTruncatedModelAndInvalidCounts();

    if (failures != 0) {
        std::cerr
            << failures
            << " VAE segment pack test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA VAE segment pack tests passed\n";
    return 0;
}
