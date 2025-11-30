#ifndef _KERNELS_INTERFACE_Int4InputsInt4Weights_H_
#define _KERNELS_INTERFACE_Int4InputsInt4Weights_H_
#include "../common/types.h"

namespace LowPrecision{
    namespace FullyConnected{
        namespace Int4InputsInt4Weights {
            int8_t* PaddingWeightsIfNeeded(const int8_t* weight, Shape shape);
            size_t TransformFilterShape(int* shape, int n_dims);
            size_t TransformInputShape(int* shape, int n_dims);
            LowPrecision::Status QuantizeFilter(const int8_t* input, LowPrecision::Shape k_shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeFilter(const uint8_t* input, LowPrecision::Shape k_shape, uint8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const int8_t* input, LowPrecision::Shape shape, int8_t* output, LowPrecision::MemLayout layout);
            LowPrecision::Status QuantizeInput(const uint8_t* input, LowPrecision::Shape shape, uint8_t* output, LowPrecision::MemLayout layout);
            Status MultiplyInt8SingleBatch(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape
            );
            LowPrecision::Status MultiplyInt8(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape);
            LowPrecision::Status MultiplyInt8MultiBatched(
                const int8_t* input, LowPrecision::Shape input_shape,
                const int8_t* kernel, LowPrecision::Shape kernel_shape,
                int32_t* output, LowPrecision::Shape output_shape,
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
            void doMultiplication1Col(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int size);
            void doMultiplication(const int8_t* activation, 
                                    int8_t* weights, 
                                    int32_t* dst_1, int32_t* dst_2,
                                    int32_t* dst_3, int32_t* dst_4,
                                    int size);
            uint8_t quantizeAndPackBitsStep(const int8_t& input, int shift_amount);
            void InputPackingStep(uint8_t* input_u, uint8_t* output, long long int size, long long int stride);
            void FilterPackingStep(uint8_t* input_u, uint8_t* output, long long int size, long long int stride);
            LowPrecision::PreprocessType InputPreProcess();
            LowPrecision::PreprocessType FilterPreProcess();
            LowPrecision::PreprocessType OutputPreProcess();
            LowPrecision::PreprocessType OutputPostProcess();
            LowPrecision::GEMMType GEMMSupport();
        }
    }
}

#endif // _KERNELS_INTERFACE_Int4InputsInt4Weights_H_