#ifndef _KERNELS_INTERFACE_Int8InputsInt4PowerWeights_H_
#define _KERNELS_INTERFACE_Int8InputsInt4PowerWeights_H_
#include "../common/types.h"

namespace LowPrecision{
    namespace FullyConnected{
        namespace Int8InputsInt4PowerWeights {
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, Shape input_shape,
                const int8_t* kernel, Shape kernel_shape,
                int32_t* output, Shape output_shape);
        }
    }
}

#endif // _KERNELS_INTERFACE_Int8InputsInt4PowerWeights_H_