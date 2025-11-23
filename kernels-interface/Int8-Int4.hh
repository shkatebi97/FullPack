#ifndef _KERNELS_INTERFACE_Int4_H_
#define _KERNELS_INTERFACE_Int4_H_
#include "../common/types.h"

namespace LowPrecision{
    namespace FullyConnected{
        namespace Int4 {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            Status PaddingWeightsIfNeeded(const int8_t* input_weight, int8_t* output_weight, Shape shape);
            Status PaddingInputsIfNeeded(const int8_t* input, int8_t* output, Shape shape);
            Shape GetPaddedShape(const Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilterWithPadding(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInputWithPadding(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status MultiplyInt8(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
                LowPrecision::MulParams params = LowPrecision::MulParams()
            );
            LowPrecision::Status MultiplyInt8MultiBatchedBlock(
                const int8_t* input, const int8_t* kernel,
                int32_t* output, const LowPrecision::Params params);
            void doMultiplication1Col(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int size);
            void doMultiplication(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int32_t* dst_2,
                                    int32_t* dst_3, int32_t* dst_4,
                                    int size);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
        }
    }
}

#endif // _KERNELS_INTERFACE_Int4_H_