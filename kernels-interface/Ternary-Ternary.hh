#ifndef _KERNELS_INTERFACE_TernaryInputsTernaryWeights_H_
#define _KERNELS_INTERFACE_TernaryInputsTernaryWeights_H_
#include "../common/types.h"

namespace LowPrecision{
    namespace FullyConnected{
        namespace TernaryInputsTernaryWeights {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const uint8_t* input, LowPrecision::Shape input_shape,
                const uint8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const LowPrecision::Params params);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
    }
}

#endif // _KERNELS_INTERFACE_TernaryInputsTernaryWeights_H_