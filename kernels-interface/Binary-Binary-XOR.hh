#ifndef _KERNELS_INTERFACE_BinaryInputsBinaryWeightsXOR_H_
#define _KERNELS_INTERFACE_BinaryInputsBinaryWeightsXOR_H_
#include "../common/types.h"

namespace LowPrecision{
    namespace FullyConnected{
        namespace BinaryInputsBinaryWeightsXOR {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
        
    }
}

#endif // _KERNELS_INTERFACE_BinaryInputsBinaryWeightsXOR_H_