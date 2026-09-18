#pragma once

#include <cstddef>
#include <cstdint>

#if KIRA_HAS_MNN
#include <MNN/Tensor.hpp>
#endif

namespace kira::pisa {

#if KIRA_HAS_MNN

enum class TensorIoScalarType {
    FLOAT16,
    FLOAT32,
    INT32,
    INT64,
    UNSUPPORTED,
};

TensorIoScalarType tensorScalarType(const MNN::Tensor* tensor);

bool writeFloatNchwTensor(
    MNN::Tensor* tensor,
    const float* values,
    std::size_t count
);

bool readFloatNchwTensor(
    const MNN::Tensor* tensor,
    float* values,
    std::size_t count
);

bool writeIntScalarTensor(
    MNN::Tensor* tensor,
    std::int64_t value
);

bool writeFp16BytesNchwTensor(
    MNN::Tensor* tensor,
    const std::uint8_t* bytes,
    std::size_t byteCount
);

#endif

}  // namespace kira::pisa
