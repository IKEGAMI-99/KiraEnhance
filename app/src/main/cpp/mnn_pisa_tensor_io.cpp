#include "mnn_pisa_tensor_io.h"

#if KIRA_HAS_MNN

#include "pisa_half.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace kira::pisa {
namespace {

using HostTensorPtr = std::unique_ptr<MNN::Tensor>;

bool logicalElementCount(
    const MNN::Tensor* tensor,
    std::size_t& output
) {
    if (tensor == nullptr) {
        return false;
    }

    const std::vector<int> shape = tensor->shape();
    if (shape.empty()) {
        output = 1;
        return true;
    }

    std::size_t count = 1;
    for (int dimension : shape) {
        if (dimension <= 0) {
            return false;
        }
        const std::size_t value =
            static_cast<std::size_t>(dimension);
        if (
            count >
            std::numeric_limits<std::size_t>::max() / value
        ) {
            return false;
        }
        count *= value;
    }

    output = count;
    return true;
}

HostTensorPtr makeNchwHostTensor(const MNN::Tensor* tensor) {
    if (tensor == nullptr) {
        return nullptr;
    }
    return std::make_unique<MNN::Tensor>(
        tensor,
        MNN::Tensor::CAFFE,
        true
    );
}

bool copyHostToTensor(
    MNN::Tensor* tensor,
    const MNN::Tensor* host
) {
    if (tensor == nullptr || host == nullptr) {
        return false;
    }
    return tensor->copyFromHostTensor(host);
}

bool copyTensorToHost(
    const MNN::Tensor* tensor,
    MNN::Tensor* host
) {
    if (tensor == nullptr || host == nullptr) {
        return false;
    }
    return tensor->copyToHostTensor(host);
}

bool validateCount(
    const MNN::Tensor* tensor,
    std::size_t expectedCount
) {
    std::size_t actualCount = 0;
    return logicalElementCount(tensor, actualCount) &&
        actualCount == expectedCount;
}

}  // namespace

TensorIoScalarType tensorScalarType(const MNN::Tensor* tensor) {
    if (tensor == nullptr) {
        return TensorIoScalarType::UNSUPPORTED;
    }

    const halide_type_t type = tensor->getType();
    if (type.lanes != 1) {
        return TensorIoScalarType::UNSUPPORTED;
    }
    if (type.code == halide_type_float && type.bits == 16) {
        return TensorIoScalarType::FLOAT16;
    }
    if (type.code == halide_type_float && type.bits == 32) {
        return TensorIoScalarType::FLOAT32;
    }
    if (type.code == halide_type_int && type.bits == 32) {
        return TensorIoScalarType::INT32;
    }
    if (type.code == halide_type_int && type.bits == 64) {
        return TensorIoScalarType::INT64;
    }
    return TensorIoScalarType::UNSUPPORTED;
}

bool writeFloatNchwTensor(
    MNN::Tensor* tensor,
    const float* values,
    std::size_t count
) {
    if (
        tensor == nullptr ||
        values == nullptr ||
        count == 0 ||
        !validateCount(tensor, count)
    ) {
        return false;
    }

    HostTensorPtr host = makeNchwHostTensor(tensor);
    if (!host) {
        return false;
    }

    switch (tensorScalarType(host.get())) {
        case TensorIoScalarType::FLOAT32: {
            float* destination = host->host<float>();
            if (destination == nullptr) {
                return false;
            }
            for (std::size_t index = 0; index < count; ++index) {
                destination[index] = values[index];
            }
            break;
        }
        case TensorIoScalarType::FLOAT16: {
            auto* destination = host->host<std::uint16_t>();
            if (destination == nullptr) {
                return false;
            }
            if (!floatsToHalfs(values, destination, count)) {
                return false;
            }
            break;
        }
        default:
            return false;
    }

    return copyHostToTensor(tensor, host.get());
}

bool readFloatNchwTensor(
    const MNN::Tensor* tensor,
    float* values,
    std::size_t count
) {
    if (
        tensor == nullptr ||
        values == nullptr ||
        count == 0 ||
        !validateCount(tensor, count)
    ) {
        return false;
    }

    HostTensorPtr host = makeNchwHostTensor(tensor);
    if (!host || !copyTensorToHost(tensor, host.get())) {
        return false;
    }

    switch (tensorScalarType(host.get())) {
        case TensorIoScalarType::FLOAT32: {
            const float* source = host->host<float>();
            if (source == nullptr) {
                return false;
            }
            for (std::size_t index = 0; index < count; ++index) {
                values[index] = source[index];
            }
            return true;
        }
        case TensorIoScalarType::FLOAT16: {
            const auto* source = host->host<std::uint16_t>();
            if (source == nullptr) {
                return false;
            }
            return halfsToFloats(source, values, count);
        }
        default:
            return false;
    }
}

bool writeIntScalarTensor(
    MNN::Tensor* tensor,
    std::int64_t value
) {
    if (tensor == nullptr || !validateCount(tensor, 1)) {
        return false;
    }

    HostTensorPtr host = makeNchwHostTensor(tensor);
    if (!host) {
        return false;
    }

    switch (tensorScalarType(host.get())) {
        case TensorIoScalarType::INT64: {
            auto* destination = host->host<std::int64_t>();
            if (destination == nullptr) {
                return false;
            }
            destination[0] = value;
            break;
        }
        case TensorIoScalarType::INT32: {
            if (
                value < std::numeric_limits<std::int32_t>::min() ||
                value > std::numeric_limits<std::int32_t>::max()
            ) {
                return false;
            }
            auto* destination = host->host<std::int32_t>();
            if (destination == nullptr) {
                return false;
            }
            destination[0] = static_cast<std::int32_t>(value);
            break;
        }
        default:
            return false;
    }

    return copyHostToTensor(tensor, host.get());
}

bool writeFp16BytesNchwTensor(
    MNN::Tensor* tensor,
    const std::uint8_t* bytes,
    std::size_t byteCount
) {
    if (
        tensor == nullptr ||
        bytes == nullptr ||
        byteCount == 0 ||
        (byteCount % 2) != 0
    ) {
        return false;
    }

    const std::size_t count = byteCount / 2;
    if (!validateCount(tensor, count)) {
        return false;
    }

    HostTensorPtr host = makeNchwHostTensor(tensor);
    if (!host) {
        return false;
    }

    switch (tensorScalarType(host.get())) {
        case TensorIoScalarType::FLOAT16: {
            auto* destination = host->host<std::uint16_t>();
            if (destination == nullptr) {
                return false;
            }
            for (std::size_t index = 0; index < count; ++index) {
                destination[index] =
                    static_cast<std::uint16_t>(bytes[index * 2]) |
                    (
                        static_cast<std::uint16_t>(
                            bytes[index * 2 + 1]
                        ) << 8U
                    );
            }
            break;
        }
        case TensorIoScalarType::FLOAT32: {
            float* destination = host->host<float>();
            if (destination == nullptr) {
                return false;
            }
            for (std::size_t index = 0; index < count; ++index) {
                const std::uint16_t half =
                    static_cast<std::uint16_t>(bytes[index * 2]) |
                    (
                        static_cast<std::uint16_t>(
                            bytes[index * 2 + 1]
                        ) << 8U
                    );
                destination[index] = halfToFloat(half);
            }
            break;
        }
        default:
            return false;
    }

    return copyHostToTensor(tensor, host.get());
}

}  // namespace kira::pisa

#endif
